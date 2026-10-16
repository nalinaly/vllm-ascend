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
  // pto: %small_acc_inline278__rv_v2
  Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v68 = Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v38, v38);
  // pto: %small_acc_inline278__rv_v2
  uint64_t v69 = (uint64_t) v34;
  TASSIGN(v68, v69);
  const int32_t v70 = 0;
  auto v71 = TPipe<0, Direction::DIR_V2C, 65536, 2, 2, false>(v13, v70, v33);
  // pto: %worker_inline203__ssa_v0
  // pto: %linear_workers_inline187__ssa_v0_idx
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
    // pto: %small_acc_inline278__ssa_v0
    ;
    Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v78 = Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v38, v38);
    // pto: %small_acc_inline278__ssa_v0
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
    // pto: %small_weight_inline211__ssa_v0
    ;
    Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v83 = Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v40, v47);
    // pto: %small_weight_inline211__ssa_v0
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
    // pto: %small_weight_t_inline182__ssa_v0
    ;
    Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v99 = Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v47, v40);
    // pto: %small_weight_t_inline182__ssa_v0
    ;
    uint64_t v100 = (uint64_t) v35;
    TASSIGN(v99, v100);
    // pto: %small_gathered_inline82__ssa_v0
    ;
    const int32_t v101 = 32;
    // pto: %small_gathered_inline82__ssa_v0
    ;
    const int32_t v102 = 512;
    // pto: %small_gathered_inline82__ssa_v0
    ;
    Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v103 = Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v101, v102);
    TPOP<TPipe<0, Direction::DIR_V2C, 65536, 2, 2, false>, Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v71, v103);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    // pto: %small_left_inline198__ssa_v0
    ;
    Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v104 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v38, v49);
    // pto: %small_left_inline198__ssa_v0
    ;
    uint64_t v105 = (uint64_t) v34;
    TASSIGN(v104, v105);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    uint16_t v106 = static_cast<uint16_t>(v48);
    uint16_t v107 = static_cast<uint16_t>(v48);
    TEXTRACT(v104, v103, v106, v107);
    // pto: %small_right_inline189__ssa_v0
    ;
    Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v108 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v49, v40);
    // pto: %small_right_inline189__ssa_v0
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
  // pto: %c49280_i64
  const int64_t v38 = 49280;
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
  // pto: %c1024_i64
  const int64_t v44 = 1024;
  // pto: %c1376_i64
  const int64_t v45 = 1376;
  const int64_t v46 = 1088;
  const int64_t v47 = 32;
  // pto: %c512_i64
  const int64_t v48 = 512;
  // pto: %c768_i64
  const int64_t v49 = 768;
  // pto: %c1280_i64
  const int64_t v50 = 1280;
  // pto: %c128_i64
  const int64_t v51 = 128;
  // pto: %c256_i64
  const int64_t v52 = 256;
  // pto: %c384_i64
  const int64_t v53 = 384;
  // pto: %c640_i64
  const int64_t v54 = 640;
  // pto: %c1792_i64
  const int64_t v55 = 1792;
  // pto: %c1824_i64
  const int64_t v56 = 1824;
  // pto: %c1856_i64
  const int64_t v57 = 1856;
  // pto: %c1888_i64
  const int64_t v58 = 1888;
  const int64_t v59 = 1056;
  // pto: %c32768_i64
  const int64_t v60 = 32768;
  // pto: %c32800_i64
  const int64_t v61 = 32800;
  // pto: %c32832_i64
  const int64_t v62 = 32832;
  const int64_t v63 = 1120;
  // pto: %c32864_i64
  const int64_t v64 = 32864;
  // pto: %c16384_i64
  const int64_t v65 = 16384;
  // pto: %c65664_i64
  const int64_t v66 = 65664;
  // pto: %c98432_i64
  const int64_t v67 = 98432;
  // pto: %c131200_i64
  const int64_t v68 = 131200;
  // pto: %c41088_i64
  const int64_t v69 = 41088;
  // pto: %c32_index
  const int64_t v70 = 32;
  // pto: %c1_index
  const int64_t v71 = 1;
  // pto: %c24_index
  const int64_t v72 = 24;
  // pto: %c16384_index
  const int64_t v73 = 16384;
  // pto: %c3_index
  const int64_t v74 = 3;
  // pto: %c11_index
  const int64_t v75 = 11;
  // pto: %c16_index
  const int64_t v76 = 16;
  // pto: %c4096_index
  const int64_t v77 = 4096;
  // pto: %c0_i32
  const int32_t v78 = 0;
  // pto: %c1536_index
  const int64_t v79 = 1536;
  // pto: %cst_43
  const float v80 = 0.0f;
  // pto: %c511_index
  const int64_t v81 = 511;
  const int64_t v82 = 1024;
  // pto: %c2_index
  const int64_t v83 = 2;
  // pto: %c0_index
  const int64_t v84 = 0;
  // pto: %c512_index
  const int64_t v85 = 512;
  // pto: %c256_index
  const int64_t v86 = 256;
  // pto: %c8_index
  const int64_t v87 = 8;
  // pto: %cst_51
  const float v88 = 6.10351563E-5f;
  // pto: %cst_52
  const float v89 = 9.99999997E-7f;
  // pto: %c192_index
  const int64_t v90 = 192;
  // pto: %cst_54
  const float v91 = 1.0f;
  // pto: %cst_55
  const float v92 = 0.5f;
  // pto: %cst_56
  const float v93 = 2.0f;
  // pto: %cst_57
  const float v94 = 0.25f;
  // pto: %cst_58
  const float v95 = 4.0f;
  // pto: %cst_59
  const float v96 = 0.125f;
  // pto: %cst_60
  const float v97 = 8.0f;
  // pto: %cst_61
  const float v98 = 0.0625f;
  // pto: %cst_62
  const float v99 = 16.0f;
  // pto: %c4_index
  const int64_t v100 = 4;
  // pto: %c5_index
  const int64_t v101 = 5;
  // pto: %c6_index
  const int64_t v102 = 6;
  // pto: %c7_index
  const int64_t v103 = 7;
  // pto: %c9_index
  const int64_t v104 = 9;
  // pto: %c10_index
  const int64_t v105 = 10;
  // pto: %c12_index
  const int64_t v106 = 12;
  // pto: %c13_index
  const int64_t v107 = 13;
  // pto: %c14_index
  const int64_t v108 = 14;
  // pto: %c15_index
  const int64_t v109 = 15;
  // pto: %c19_index
  const int64_t v110 = 19;
  // pto: %c64_index
  const int64_t v111 = 64;
  // pto: %c8192_index
  const int64_t v112 = 8192;
  // pto: %c12288_index
  const int64_t v113 = 12288;
  // pto: %446
  int64_t v114 = (int64_t) ((uint64_t) v29 * (uint64_t) v75);
  // pto: %rms_partials_inline166__ssa_v0_view
  const int64_t v115 = 1;
  // pto: %rms_partials_inline166__ssa_v0_view
  const int64_t v116 = 1;
  // pto: %rms_partials_inline166__ssa_v0_view
  const int64_t v117 = 1;
  // pto: %rms_partials_inline166__ssa_v0_view
  int64_t v118 = v114 * v71;
  // pto: %rms_partials_inline166__ssa_v0_view
  int64_t v119 = v117 * v118;
  // pto: %rms_partials_inline166__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v120 = pto::Shape<1, 1, 1, -1, -1>(v115, v116, v117, v114, v71);
  // pto: %rms_partials_inline166__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v121 = pto::Stride<-1, -1, -1, -1, -1>(v116 * v119, v119, v118, v71, v114);
  // pto: %rms_partials_inline166__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v122 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v120, v121);
  // pto: %x_flat_inline122__ssa_v0_view
  const int64_t v123 = 1;
  // pto: %x_flat_inline122__ssa_v0_view
  const int64_t v124 = 1;
  // pto: %x_flat_inline122__ssa_v0_view
  const int64_t v125 = 1;
  // pto: %x_flat_inline122__ssa_v0_view
  int64_t v126 = (int64_t) v30;
  // pto: %x_flat_inline122__ssa_v0_view
  int64_t v127 = v126 * v73;
  // pto: %x_flat_inline122__ssa_v0_view
  int64_t v128 = v125 * v127;
  // pto: %x_flat_inline122__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v129 = pto::Shape<1, 1, 1, -1, -1>(v123, v124, v125, v126, v73);
  // pto: %x_flat_inline122__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v130 = pto::Stride<-1, -1, -1, -1, -1>(v124 * v128, v128, v127, v73, v71);
  // pto: %x_flat_inline122__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v131 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v129, v130);
  // pto: %447
  int64_t v132 = (int64_t) ((uint64_t) v29 * (uint64_t) v70);
  // pto: %parts_2d_inline62__ssa_v0_view
  const int64_t v133 = 1;
  // pto: %parts_2d_inline62__ssa_v0_view
  const int64_t v134 = 1;
  // pto: %parts_2d_inline62__ssa_v0_view
  const int64_t v135 = 1;
  // pto: %parts_2d_inline62__ssa_v0_view
  int64_t v136 = v75 * v132;
  // pto: %parts_2d_inline62__ssa_v0_view
  int64_t v137 = v135 * v136;
  // pto: %parts_2d_inline62__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v138 = pto::Shape<1, 1, 1, -1, -1>(v133, v134, v135, v75, v132);
  // pto: %parts_2d_inline62__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v139 = pto::Stride<-1, -1, -1, -1, -1>(v134 * v137, v137, v136, v132, v71);
  // pto: %parts_2d_inline62__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v140 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v138, v139);
  // pto: %rms_2d_inline255__ssa_v0_view
  const int64_t v141 = 1;
  // pto: %rms_2d_inline255__ssa_v0_view
  const int64_t v142 = 1;
  // pto: %rms_2d_inline255__ssa_v0_view
  const int64_t v143 = 1;
  // pto: %rms_2d_inline255__ssa_v0_view
  int64_t v144 = (int64_t) v29;
  // pto: %rms_2d_inline255__ssa_v0_view
  int64_t v145 = v75 * v144;
  // pto: %rms_2d_inline255__ssa_v0_view
  int64_t v146 = v143 * v145;
  // pto: %rms_2d_inline255__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v147 = pto::Shape<1, 1, 1, -1, -1>(v141, v142, v143, v75, (int64_t) v29);
  // pto: %rms_2d_inline255__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v148 = pto::Stride<-1, -1, -1, -1, -1>(v142 * v146, v146, v145, v144, v71);
  // pto: %rms_2d_inline255__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v149 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v147, v148);
  // pto: %hc_base_2d_inline286__ssa_v0_view
  const int64_t v150 = 1;
  // pto: %hc_base_2d_inline286__ssa_v0_view
  const int64_t v151 = 1;
  // pto: %hc_base_2d_inline286__ssa_v0_view
  const int64_t v152 = 1;
  // pto: %hc_base_2d_inline286__ssa_v0_view
  int64_t v153 = v71 * v72;
  // pto: %hc_base_2d_inline286__ssa_v0_view
  int64_t v154 = v152 * v153;
  // pto: %hc_base_2d_inline286__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v155 = pto::Shape<1, 1, 1, -1, -1>(v150, v151, v152, v71, v72);
  // pto: %hc_base_2d_inline286__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v156 = pto::Stride<-1, -1, -1, -1, -1>(v151 * v154, v154, v153, v72, v71);
  // pto: %hc_base_2d_inline286__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v157 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v8, v155, v156);
  // pto: %comb_inline153__ssa_v0_view
  const int64_t v158 = 1;
  // pto: %comb_inline153__ssa_v0_view
  const int64_t v159 = 1;
  // pto: %comb_inline153__ssa_v0_view
  const int64_t v160 = 1;
  // pto: %comb_inline153__ssa_v0_view
  int64_t v161 = (int64_t) v30;
  // pto: %comb_inline153__ssa_v0_view
  int64_t v162 = v161 * v76;
  // pto: %comb_inline153__ssa_v0_view
  int64_t v163 = v160 * v162;
  // pto: %comb_inline153__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v164 = pto::Shape<1, 1, 1, -1, -1>(v158, v159, v160, v161, v76);
  // pto: %comb_inline153__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v165 = pto::Stride<-1, -1, -1, -1, -1>(v159 * v163, v163, v162, v76, v71);
  // pto: %comb_inline153__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v166 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v9, v164, v165);
  // pto: %hc_scale_2d_inline132__ssa_v0_view
  const int64_t v167 = 1;
  // pto: %hc_scale_2d_inline132__ssa_v0_view
  const int64_t v168 = 1;
  // pto: %hc_scale_2d_inline132__ssa_v0_view
  const int64_t v169 = 1;
  // pto: %hc_scale_2d_inline132__ssa_v0_view
  int64_t v170 = v71 * v74;
  // pto: %hc_scale_2d_inline132__ssa_v0_view
  int64_t v171 = v169 * v170;
  // pto: %hc_scale_2d_inline132__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v172 = pto::Shape<1, 1, 1, -1, -1>(v167, v168, v169, v71, v74);
  // pto: %hc_scale_2d_inline132__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v173 = pto::Stride<-1, -1, -1, -1, -1>(v168 * v171, v171, v170, v74, v71);
  // pto: %hc_scale_2d_inline132__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v174 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v10, v172, v173);
  // pto: %448
  int64_t v175 = (int64_t) ((uint64_t) v30 * (uint64_t) v100);
  // pto: %post_flat_inline228__ssa_v0_view
  const int64_t v176 = 1;
  // pto: %post_flat_inline228__ssa_v0_view
  const int64_t v177 = 1;
  // pto: %post_flat_inline228__ssa_v0_view
  const int64_t v178 = 1;
  // pto: %post_flat_inline228__ssa_v0_view
  int64_t v179 = v71 * v175;
  // pto: %post_flat_inline228__ssa_v0_view
  int64_t v180 = v178 * v179;
  // pto: %post_flat_inline228__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v181 = pto::Shape<1, 1, 1, -1, -1>(v176, v177, v178, v71, v175);
  // pto: %post_flat_inline228__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v182 = pto::Stride<-1, -1, -1, -1, -1>(v177 * v180, v180, v179, v175, v71);
  // pto: %post_flat_inline228__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v183 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v11, v181, v182);
  // pto: %mixed_view_inline142__ssa_v0_view
  const int64_t v184 = 1;
  // pto: %mixed_view_inline142__ssa_v0_view
  const int64_t v185 = 1;
  // pto: %mixed_view_inline142__ssa_v0_view
  const int64_t v186 = 1;
  // pto: %mixed_view_inline142__ssa_v0_view
  int64_t v187 = (int64_t) v30;
  // pto: %mixed_view_inline142__ssa_v0_view
  int64_t v188 = v187 * v77;
  // pto: %mixed_view_inline142__ssa_v0_view
  int64_t v189 = v186 * v188;
  // pto: %mixed_view_inline142__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v190 = pto::Shape<1, 1, 1, -1, -1>(v184, v185, v186, v187, v77);
  // pto: %mixed_view_inline142__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v191 = pto::Stride<-1, -1, -1, -1, -1>(v185 * v189, v189, v188, v77, v71);
  // pto: %mixed_view_inline142__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v192 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v12, v190, v191);
  // pto: %small_rms_total_inline181__rv_v2
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v193 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v76);
  // pto: %small_rms_total_inline181__rv_v2
  uint64_t v194 = (uint64_t) v40;
  TASSIGN(v193, v194);
  const int32_t v195 = 0;
  auto v196 = TPipe<0, Direction::DIR_V2C, 65536, 2, 2, false>(v13, v195, v78);
  // pto: %worker_inline203__ssa_v0
  int64_t v197 = (int64_t) v31;
  // pto: %linear_workers_inline187__ssa_v0_idx
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  for (int64_t v198 = v197; v198 < v14; v198 += (int64_t) v15) {
    // pto: %54, %55
    ;
    int64_t v199 = (int64_t) ((uint64_t) (v198 / v75) * (uint64_t) v70);
    // pto: %56
    ;
    int64_t v200 = v198 % v75;
    // pto: %57
    ;
    int64_t v201 = (int64_t) ((uint64_t) v200 * (uint64_t) v79);
    // pto: %58
    ;
    int64_t v202 = (int64_t) ((uint64_t) v73 - (uint64_t) v201);
    // pto: %59
    ;
    int64_t v203 = v202 < v79 ? v202 : v79;
    // pto: %small_rms_total_inline181__ssa_v0
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v204 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v76);
    // pto: %small_rms_total_inline181__ssa_v0
    ;
    uint64_t v205 = (uint64_t) v37;
    TASSIGN(v204, v205);
    pipe_barrier(PIPE_V);
    TEXPANDS(v204, v80);
    // pto: %62
    ;
    int64_t v206 = (int64_t) ((uint64_t) v203 + (uint64_t) v81);
    // pto: %63
    ;
    int64_t v207 = v206 / v82;
    // pto: %small_lane_inline185__ssa_v0, %65
    ;
    int64_t v208 = (int64_t) ((uint64_t) ((int64_t) v33) * (uint64_t) v76);
    // pto: %66
    ;
    int64_t v209 = (int64_t) ((uint64_t) v199 + (uint64_t) v208);
    // pto: %67
    ;
    int64_t v210 = (int64_t) ((uint64_t) v17 - (uint64_t) v209);
    // pto: %68
    ;
    int64_t v211 = v210 < v76 ? v210 : v76;
    // pto: %69
    ;
    int64_t v212 = v211 < v84 ? v84 : v211;
    // pto: %70
    ;
    int64_t v213 = (int64_t) ((uint64_t) v201 + (uint64_t) v85);
    // pto: %small_input_inline264__ssa_v0
    ;
    Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v214 = Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v212, v85);
    // pto: %small_input_inline264__ssa_v0
    ;
    uint64_t v215 = (uint64_t) v38;
    TASSIGN(v214, v215);
    // pto: %77
    ;
    int64_t v216 = v209 < v84 ? v84 : v209;
    // pto: %78
    ;
    int64_t v217 = v201 < v84 ? v84 : v201;
    // pto: %x_flat_inline122__ssa_v0_pview
    ;
    const int64_t v218 = 0;
    // pto: %x_flat_inline122__ssa_v0_pview
    ;
    __gm__ bfloat16_t* v219 = PTOAS__GLOBAL_TENSOR_DATA(v131);
    // pto: %x_flat_inline122__ssa_v0_pview
    ;
    const int64_t v220 = 1;
    // pto: %x_flat_inline122__ssa_v0_pview
    ;
    const int64_t v221 = 1;
    // pto: %x_flat_inline122__ssa_v0_pview
    ;
    const int64_t v222 = 1;
    // pto: %x_flat_inline122__ssa_v0_pview
    ;
    int64_t v223 = v212 * v73;
    // pto: %x_flat_inline122__ssa_v0_pview
    ;
    int64_t v224 = v222 * v223;
    // pto: %x_flat_inline122__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, -1, 512> v225 = pto::Shape<1, 1, 1, -1, 512>(v220, v221, v222, v212, v85);
    // pto: %x_flat_inline122__ssa_v0_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v226 = pto::Stride<-1, -1, -1, -1, -1>(v221 * v224, v224, v223, v73, v71);
    // pto: %x_flat_inline122__ssa_v0_pview
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v227 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v219 + (v218 + v216 * v73 + v217 * v71), v225, v226);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    TLOAD(v214, v227);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %0
    ;
    Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v228 = Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v212, v85);
    // pto: %0
    ;
    uint64_t v229 = (uint64_t) v39;
    TASSIGN(v228, v229);
    // pto: %80
    ;
    int64_t v230 = v213 < v84 ? v84 : v213;
    // pto: %81
    ;
    const int64_t v231 = 0;
    // pto: %81
    ;
    __gm__ bfloat16_t* v232 = PTOAS__GLOBAL_TENSOR_DATA(v131);
    // pto: %81
    ;
    const int64_t v233 = 1;
    // pto: %81
    ;
    const int64_t v234 = 1;
    // pto: %81
    ;
    const int64_t v235 = 1;
    // pto: %81
    ;
    int64_t v236 = v212 * v73;
    // pto: %81
    ;
    int64_t v237 = v235 * v236;
    // pto: %81
    ;
    pto::Shape<1, 1, 1, -1, 512> v238 = pto::Shape<1, 1, 1, -1, 512>(v233, v234, v235, v212, v85);
    // pto: %81
    ;
    pto::Stride<-1, -1, -1, -1, -1> v239 = pto::Stride<-1, -1, -1, -1, -1>(v234 * v237, v237, v236, v73, v71);
    // pto: %81
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v240 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v232 + (v231 + v216 * v73 + v230 * v71), v238, v239);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    TLOAD(v228, v240);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %small_fp32_inline173__ssa_v0
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v241 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v212, v85);
    // pto: %small_fp32_inline173__ssa_v0
    ;
    uint64_t v242 = (uint64_t) v40;
    TASSIGN(v241, v242);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    RoundMode v243 = RoundMode::CAST_ROUND;
    SaturationMode v244 = SaturationMode::ON;
    TCVT(v241, v214, v243, v244);
    v241.SetValidShape(v212, v85);
    // pto: %small_clean_inline110__ssa_v0
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v245 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v76, v85);
    // pto: %small_clean_inline110__ssa_v0
    ;
    uint64_t v246 = (uint64_t) v40;
    TASSIGN(v245, v246);
    pipe_barrier(PIPE_V);
    TFILLPAD<pto::TFillPadMode::InPlace>(v245, v241);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    v245.SetValidShape(v76, v85);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TPUSH<TPipe<0, Direction::DIR_V2C, 65536, 2, 2, false>, Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v196, v245);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    // pto: %small_rms_tmp_inline200__ssa_v0
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v247 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v85);
    // pto: %small_rms_tmp_inline200__ssa_v0
    ;
    uint64_t v248 = (uint64_t) v38;
    TASSIGN(v247, v248);
    // pto: %t__tmp_v1
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v249 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v76, v85);
    // pto: %t__tmp_v1
    ;
    uint64_t v250 = (uint64_t) v40;
    TASSIGN(v249, v250);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    TMUL(v249, v245, v245);
    // pto: %small_sq_inline208__ssa_v0
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v251 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v71);
    // pto: %small_sq_inline208__ssa_v0
    ;
    uint64_t v252 = (uint64_t) v41;
    TASSIGN(v251, v252);
    pipe_barrier(PIPE_V);
    TROWSUM(v251, v249, v247);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    // pto: %t__tmp_v2
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v253 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v76);
    // pto: %t__tmp_v2
    ;
    uint64_t v254 = (uint64_t) v41;
    TASSIGN(v253, v254);
    // pto: %small_rms_total_inline181__ssa_v3
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v255 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v76);
    // pto: %small_rms_total_inline181__ssa_v3
    ;
    uint64_t v256 = (uint64_t) v40;
    TASSIGN(v255, v256);
    pipe_barrier(PIPE_V);
    TADD(v255, v204, v253);
    // pto: %1
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v257 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v212, v85);
    // pto: %1
    ;
    uint64_t v258 = (uint64_t) v42;
    TASSIGN(v257, v258);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    RoundMode v259 = RoundMode::CAST_ROUND;
    SaturationMode v260 = SaturationMode::ON;
    TCVT(v257, v228, v259, v260);
    v257.SetValidShape(v212, v85);
    // pto: %3
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v261 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v76, v85);
    // pto: %3
    ;
    uint64_t v262 = (uint64_t) v42;
    TASSIGN(v261, v262);
    pipe_barrier(PIPE_V);
    TFILLPAD<pto::TFillPadMode::InPlace>(v261, v257);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    v261.SetValidShape(v76, v85);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TPUSH<TPipe<0, Direction::DIR_V2C, 65536, 2, 2, false>, Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v196, v261);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    // pto: %5
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v263 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v85);
    // pto: %5
    ;
    uint64_t v264 = (uint64_t) v39;
    TASSIGN(v263, v264);
    // pto: %6
    ;
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v265 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v76, v85);
    // pto: %6
    ;
    uint64_t v266 = (uint64_t) v42;
    TASSIGN(v265, v266);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    TMUL(v265, v261, v261);
    // pto: %7
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v267 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v71);
    // pto: %7
    ;
    uint64_t v268 = (uint64_t) v43;
    TASSIGN(v267, v268);
    pipe_barrier(PIPE_V);
    TROWSUM(v267, v265, v263);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    // pto: %8
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v269 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v76);
    // pto: %8
    ;
    uint64_t v270 = (uint64_t) v43;
    TASSIGN(v269, v270);
    // pto: %9
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v271 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v76);
    // pto: %9
    ;
    uint64_t v272 = (uint64_t) v37;
    TASSIGN(v271, v272);
    pipe_barrier(PIPE_V);
    TADD(v271, v255, v269);
    // pto: %83, %84, %64, %85
    ;
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    if ((int64_t) ((uint64_t) (v206 / v85) - (uint64_t) ((int64_t) (uint64_t) v207 * (uint64_t) v83)) == v71) {
      // pto: %87
      ;
      int64_t v273 = (int64_t) ((uint64_t) v201 + (uint64_t) ((int64_t) (uint64_t) v207 * (uint64_t) v82));
      // pto: %10
      ;
      Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v274 = Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v212, v85);
      // pto: %10
      ;
      uint64_t v275 = (uint64_t) v38;
      TASSIGN(v274, v275);
      // pto: %95
      ;
      int64_t v276 = v273 < v84 ? v84 : v273;
      // pto: %96
      ;
      const int64_t v277 = 0;
      // pto: %96
      ;
      __gm__ bfloat16_t* v278 = PTOAS__GLOBAL_TENSOR_DATA(v131);
      // pto: %96
      ;
      const int64_t v279 = 1;
      // pto: %96
      ;
      const int64_t v280 = 1;
      // pto: %96
      ;
      const int64_t v281 = 1;
      // pto: %96
      ;
      int64_t v282 = v212 * v73;
      // pto: %96
      ;
      int64_t v283 = v281 * v282;
      // pto: %96
      ;
      pto::Shape<1, 1, 1, -1, 512> v284 = pto::Shape<1, 1, 1, -1, 512>(v279, v280, v281, v212, v85);
      // pto: %96
      ;
      pto::Stride<-1, -1, -1, -1, -1> v285 = pto::Stride<-1, -1, -1, -1, -1>(v280 * v283, v283, v282, v73, v71);
      // pto: %96
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v286 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v278 + (v277 + v216 * v73 + v276 * v71), v284, v285);
      TLOAD(v274, v286);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %11
      ;
      Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v287 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v212, v85);
      // pto: %11
      ;
      uint64_t v288 = (uint64_t) v40;
      TASSIGN(v287, v288);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      pipe_barrier(PIPE_V);
      RoundMode v289 = RoundMode::CAST_ROUND;
      SaturationMode v290 = SaturationMode::ON;
      TCVT(v287, v274, v289, v290);
      v287.SetValidShape(v212, v85);
      // pto: %13
      ;
      Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v291 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v76, v85);
      // pto: %13
      ;
      uint64_t v292 = (uint64_t) v40;
      TASSIGN(v291, v292);
      pipe_barrier(PIPE_V);
      TFILLPAD<pto::TFillPadMode::InPlace>(v291, v287);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
      v291.SetValidShape(v76, v85);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
      TPUSH<TPipe<0, Direction::DIR_V2C, 65536, 2, 2, false>, Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v196, v291);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      // pto: %15
      ;
      Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v293 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v85);
      // pto: %15
      ;
      uint64_t v294 = (uint64_t) v38;
      TASSIGN(v293, v294);
      // pto: %16
      ;
      Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v295 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v76, v85);
      // pto: %16
      ;
      uint64_t v296 = (uint64_t) v40;
      TASSIGN(v295, v296);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      TMUL(v295, v291, v291);
      // pto: %17
      ;
      Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v297 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v71);
      // pto: %17
      ;
      uint64_t v298 = (uint64_t) v42;
      TASSIGN(v297, v298);
      pipe_barrier(PIPE_V);
      TROWSUM(v297, v295, v293);
      // pto: %18
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v299 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v76);
      // pto: %18
      ;
      uint64_t v300 = (uint64_t) v42;
      TASSIGN(v299, v300);
      // pto: %19
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v301 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v76);
      // pto: %19
      ;
      uint64_t v302 = (uint64_t) v40;
      TASSIGN(v301, v302);
      pipe_barrier(PIPE_V);
      TADD(v301, v271, v299);
    } else {
      // pto: %small_rms_total_inline181__rv_v2_main_mv
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v303 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v76);
      // pto: %small_rms_total_inline181__rv_v2_main_mv
      ;
      uint64_t v304 = (uint64_t) v40;
      TASSIGN(v303, v304);
      pipe_barrier(PIPE_V);
      TMOV(v303, v271);
    };
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
    // pto: %60, %61, %98
    ;
    int64_t v305 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v200 * (uint64_t) v16) + (uint64_t) v199) + (uint64_t) v208);
    // pto: %t__tmp_v3
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v306 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v71);
    // pto: %t__tmp_v3
    ;
    uint64_t v307 = (uint64_t) v40;
    TASSIGN(v306, v307);
    // pto: %99
    ;
    int64_t v308 = v305 < v84 ? v84 : v305;
    // pto: %rms_partials_inline166__iter_v1_pview
    ;
    const int64_t v309 = 0;
    // pto: %rms_partials_inline166__iter_v1_pview
    ;
    __gm__ float* v310 = PTOAS__GLOBAL_TENSOR_DATA(v122);
    // pto: %rms_partials_inline166__iter_v1_pview
    ;
    const int64_t v311 = 1;
    // pto: %rms_partials_inline166__iter_v1_pview
    ;
    const int64_t v312 = 1;
    // pto: %rms_partials_inline166__iter_v1_pview
    ;
    const int64_t v313 = 1;
    // pto: %rms_partials_inline166__iter_v1_pview
    ;
    int64_t v314 = v76 * v71;
    // pto: %rms_partials_inline166__iter_v1_pview
    ;
    int64_t v315 = v313 * v314;
    // pto: %rms_partials_inline166__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 16, 1> v316 = pto::Shape<1, 1, 1, 16, 1>(v311, v312, v313, v76, v71);
    // pto: %rms_partials_inline166__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v317 = pto::Stride<-1, -1, -1, -1, -1>(v312 * v315, v315, v314, v71, v114);
    // pto: %rms_partials_inline166__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v318 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v310 + (v309 + v308 * v71 + v84 * v114), v316, v317);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
    TSTORE(v318, v306);
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
  // pto: %100, %101, %tail_lane_inline111__ssa_v0
  int64_t v319 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v197 * (uint64_t) v83) + (uint64_t) ((int64_t) v33));
  // pto: %102
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  if (v319 < v18) {
    // pto: %104
    ;
    int64_t v320 = v319 < v20 ? v319 : v20;
    // pto: %103, %105
    ;
    int64_t v321 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v319 * (uint64_t) v19) + (uint64_t) v320);
    // pto: %107
    ;
    int64_t v322 = (int64_t) ((uint64_t) v20 - (uint64_t) v319);
    // pto: %108
    ;
    int64_t v323 = v322 < v84 ? v84 : v322;
    // pto: %109
    ;
    int64_t v324 = v323 < v71 ? v323 : v71;
    // pto: %106, %110
    ;
    int64_t v325 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v321 + (uint64_t) v19) + (uint64_t) v324);
    for (int64_t v326 = v321; v326 < v325; v326 += v83) {
      // pto: %scale2_inline165__tile
      ;
      float v327 = (v5)[v83];
      // pto: %111
      ;
      int64_t v328 = (int64_t) ((uint64_t) v325 - (uint64_t) v326);
      // pto: %112
      ;
      int64_t v329 = v328 < v83 ? v328 : v83;
      // pto: %113
      ;
      int64_t v330 = (int64_t) ((uint64_t) v329 * (uint64_t) v70);
      // pto: %c_parts_load_inline141__ssa_v0
      ;
      Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v331 = Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v75, v330);
      // pto: %c_parts_load_inline141__ssa_v0
      ;
      uint64_t v332 = (uint64_t) v40;
      TASSIGN(v331, v332);
      // pto: %114
      ;
      int64_t v333 = (int64_t) ((uint64_t) v326 * (uint64_t) v70);
      // pto: %115
      ;
      int64_t v334 = v333 < v84 ? v84 : v333;
      // pto: %parts_2d_inline62__ssa_v0_pview
      ;
      const int64_t v335 = 0;
      // pto: %parts_2d_inline62__ssa_v0_pview
      ;
      __gm__ float* v336 = PTOAS__GLOBAL_TENSOR_DATA(v140);
      // pto: %parts_2d_inline62__ssa_v0_pview
      ;
      const int64_t v337 = 1;
      // pto: %parts_2d_inline62__ssa_v0_pview
      ;
      const int64_t v338 = 1;
      // pto: %parts_2d_inline62__ssa_v0_pview
      ;
      const int64_t v339 = 1;
      // pto: %parts_2d_inline62__ssa_v0_pview
      ;
      int64_t v340 = v75 * v132;
      // pto: %parts_2d_inline62__ssa_v0_pview
      ;
      int64_t v341 = v339 * v340;
      // pto: %parts_2d_inline62__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 11, -1> v342 = pto::Shape<1, 1, 1, 11, -1>(v337, v338, v339, v75, v330);
      // pto: %parts_2d_inline62__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v343 = pto::Stride<-1, -1, -1, -1, -1>(v338 * v341, v341, v340, v132, v71);
      // pto: %parts_2d_inline62__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 11, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v344 = GlobalTensor<float, pto::Shape<1, 1, 1, 11, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v336 + (v335 + v84 * v132 + v334 * v71), v342, v343);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      TLOAD(v331, v344);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      // pto: %t__tmp_v4
      ;
      Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v345 = Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v75, v86);
      // pto: %t__tmp_v4
      ;
      uint64_t v346 = (uint64_t) v40;
      TASSIGN(v345, v346);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      TFILLPAD<pto::TFillPadMode::InPlace>(v345, v331);
      v345.SetValidShape(v75, v86);
      // pto: %t__tmp_v5
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v347 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v86);
      // pto: %t__tmp_v5
      ;
      uint64_t v348 = (uint64_t) v40;
      TASSIGN(v347, v348);
      pipe_barrier(PIPE_V);
      TCOLSUM(v347, v345);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      // pto: %c_total_inline188__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v349 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v70);
      // pto: %c_total_inline188__ssa_v0
      ;
      uint64_t v350 = (uint64_t) v40;
      TASSIGN(v349, v350);
      // pto: %c_sq_load_inline36__ssa_v0
      ;
      Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v351 = Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v75, v329);
      // pto: %c_sq_load_inline36__ssa_v0
      ;
      uint64_t v352 = (uint64_t) v44;
      TASSIGN(v351, v352);
      // pto: %117
      ;
      int64_t v353 = v326 < v84 ? v84 : v326;
      // pto: %rms_2d_inline255__ssa_v0_pview
      ;
      int64_t v354 = (int64_t) v29;
      // pto: %rms_2d_inline255__ssa_v0_pview
      ;
      const int64_t v355 = 0;
      // pto: %rms_2d_inline255__ssa_v0_pview
      ;
      __gm__ float* v356 = PTOAS__GLOBAL_TENSOR_DATA(v149);
      // pto: %rms_2d_inline255__ssa_v0_pview
      ;
      const int64_t v357 = 1;
      // pto: %rms_2d_inline255__ssa_v0_pview
      ;
      const int64_t v358 = 1;
      // pto: %rms_2d_inline255__ssa_v0_pview
      ;
      const int64_t v359 = 1;
      // pto: %rms_2d_inline255__ssa_v0_pview
      ;
      int64_t v360 = v75 * v354;
      // pto: %rms_2d_inline255__ssa_v0_pview
      ;
      int64_t v361 = v359 * v360;
      // pto: %rms_2d_inline255__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 11, -1> v362 = pto::Shape<1, 1, 1, 11, -1>(v357, v358, v359, v75, v329);
      // pto: %rms_2d_inline255__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v363 = pto::Stride<-1, -1, -1, -1, -1>(v358 * v361, v361, v360, v354, v71);
      // pto: %rms_2d_inline255__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 11, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v364 = GlobalTensor<float, pto::Shape<1, 1, 1, 11, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v356 + (v355 + v84 * v354 + v353 * v71), v362, v363);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      TLOAD(v351, v364);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      // pto: %t__tmp_v6
      ;
      Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v365 = Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v75, v87);
      // pto: %t__tmp_v6
      ;
      uint64_t v366 = (uint64_t) v44;
      TASSIGN(v365, v366);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      TFILLPAD<pto::TFillPadMode::InPlace>(v365, v351);
      v365.SetValidShape(v75, v87);
      // pto: %c_rms_tmp_inline65__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v367 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %c_rms_tmp_inline65__ssa_v0
      ;
      uint64_t v368 = (uint64_t) v45;
      TASSIGN(v367, v368);
      // pto: %t__tmp_v7
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v369 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %t__tmp_v7
      ;
      uint64_t v370 = (uint64_t) v44;
      TASSIGN(v369, v370);
      pipe_barrier(PIPE_V);
      TCOLSUM(v369, v365);
      // pto: %t__tmp_v8
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v371 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %t__tmp_v8
      ;
      uint64_t v372 = (uint64_t) v44;
      TASSIGN(v371, v372);
      pipe_barrier(PIPE_V);
      TMULS(v371, v369, v88);
      // pto: %t__tmp_v9
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v373 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %t__tmp_v9
      ;
      uint64_t v374 = (uint64_t) v44;
      TASSIGN(v373, v374);
      pipe_barrier(PIPE_V);
      TADDS(v373, v371, v89);
      // pto: %t__tmp_v10
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v375 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %t__tmp_v10
      ;
      uint64_t v376 = (uint64_t) v46;
      TASSIGN(v375, v376);
      pipe_barrier(PIPE_V);
      TRSQRT(v375, v373, v367);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      // pto: %c_inv_inline34__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v377 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v71);
      // pto: %c_inv_inline34__ssa_v0
      ;
      uint64_t v378 = (uint64_t) v46;
      TASSIGN(v377, v378);
      // pto: %comb_inv_inline51__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v379 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v71);
      // pto: %comb_inv_inline51__ssa_v0
      ;
      uint64_t v380 = (uint64_t) v46;
      TASSIGN(v379, v380);
      // pto: %t__tmp_v11
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v381 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v76);
      // pto: %t__tmp_v11
      ;
      uint64_t v382 = (uint64_t) v47;
      TASSIGN(v381, v382);
      // pto: %slice_view
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 16, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v383;
      // pto: %slice_view
      ;
      uint64_t v384 = (uint64_t) v47;
      TASSIGN(v383, v384);
      // pto: %raw_inline119__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v385 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v76);
      // pto: %raw_inline119__ssa_v0
      ;
      uint64_t v386 = (uint64_t) v40;
      TASSIGN(v385, v386);
      TADDS(v385, v383, v80);
      // pto: %base_inline214__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v387 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v76);
      // pto: %base_inline214__ssa_v0
      ;
      uint64_t v388 = (uint64_t) v44;
      TASSIGN(v387, v388);
      // pto: %hc_base_2d_inline286__ssa_v0_pview
      ;
      __gm__ float* v389 = PTOAS__GLOBAL_TENSOR_DATA(v157);
      // pto: %hc_base_2d_inline286__ssa_v0_pview
      ;
      unsigned v390 = 8;
      // pto: %hc_base_2d_inline286__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 16> v391 = pto::Shape<1, 1, 1, 1, 16>();
      // pto: %hc_base_2d_inline286__ssa_v0_pview
      ;
      pto::Stride<24, 24, 24, 24, 1> v392 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %hc_base_2d_inline286__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v393 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v389 + v390, v391, v392);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      TLOAD(v387, v393);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      // pto: %t__tmp_v12
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v394 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v76);
      // pto: %t__tmp_v12
      ;
      uint64_t v395 = (uint64_t) v48;
      TASSIGN(v394, v395);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v394, v385, v379);
      // pto: %t__tmp_v13
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v396 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v76);
      // pto: %t__tmp_v13
      ;
      uint64_t v397 = (uint64_t) v48;
      TASSIGN(v396, v397);
      pipe_barrier(PIPE_V);
      TMULS(v396, v394, v327);
      // pto: %t__tmp_v14
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v398 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v76);
      // pto: %t__tmp_v14
      ;
      uint64_t v399 = (uint64_t) v40;
      TASSIGN(v398, v399);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      TCOLEXPAND(v398, v387);
      // pto: %logits_inline170__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v400 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v76);
      // pto: %logits_inline170__ssa_v0
      ;
      uint64_t v401 = (uint64_t) v49;
      TASSIGN(v400, v401);
      pipe_barrier(PIPE_V);
      TADD(v400, v396, v398);
      // pto: %t__tmp_v15
      ;
      Tile<TileType::Vec, float, 2, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v402 = Tile<TileType::Vec, float, 2, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v83, v76);
      // pto: %t__tmp_v15
      ;
      uint64_t v403 = (uint64_t) v49;
      TASSIGN(v402, v403);
      // pto: %regular_logits_inline265__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v404 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %regular_logits_inline265__ssa_v0
      ;
      uint64_t v405 = (uint64_t) v49;
      TASSIGN(v404, v405);
      // pto: %ids_inline219__ci_tmp_v0
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v406 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v90);
      // pto: %ids_inline219__ci_tmp_v0
      ;
      uint64_t v407 = (uint64_t) v40;
      TASSIGN(v406, v407);
      // pto: %ids_inline219__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v408 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %ids_inline219__ssa_v1
      ;
      uint64_t v409 = (uint64_t) v50;
      TASSIGN(v408, v409);
      // pto: %ci_tmp_view
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v410;
      TRESHAPE(v410, v406);
      // pto: %ci_dst_view
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v411;
      TRESHAPE(v411, v408);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v411, v78, v410);
      // pto: %tmp_inline93__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v412 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %tmp_inline93__ssa_v1
      ;
      uint64_t v413 = (uint64_t) v40;
      TASSIGN(v412, v413);
      // pto: %idx_float_inline77__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v414 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %idx_float_inline77__ssa_v0
      ;
      uint64_t v415 = (uint64_t) v51;
      TASSIGN(v414, v415);
      pipe_barrier(PIPE_V);
      RoundMode v416 = RoundMode::CAST_ROUND;
      SaturationMode v417 = SaturationMode::ON;
      TCVT(v414, v408, v416, v417);
      // pto: %t__tmp_v16
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v418 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v16
      ;
      uint64_t v419 = (uint64_t) v52;
      TASSIGN(v418, v419);
      pipe_barrier(PIPE_V);
      TMULS(v418, v414, v91);
      // pto: %t__tmp_v17
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v420 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v17
      ;
      uint64_t v421 = (uint64_t) v52;
      TASSIGN(v420, v421);
      pipe_barrier(PIPE_V);
      RoundMode v422 = RoundMode::CAST_FLOOR;
      SaturationMode v423 = SaturationMode::ON;
      TCVT(v420, v418, v422, v423);
      // pto: %ri1_low_inline103__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v424 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %ri1_low_inline103__ssa_v0
      ;
      uint64_t v425 = (uint64_t) v52;
      TASSIGN(v424, v425);
      pipe_barrier(PIPE_V);
      RoundMode v426 = RoundMode::CAST_ROUND;
      SaturationMode v427 = SaturationMode::ON;
      TCVT(v424, v420, v426, v427);
      // pto: %t__tmp_v18
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v428 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v18
      ;
      uint64_t v429 = (uint64_t) v53;
      TASSIGN(v428, v429);
      TMULS(v428, v414, v92);
      // pto: %t__tmp_v19
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v430 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v19
      ;
      uint64_t v431 = (uint64_t) v53;
      TASSIGN(v430, v431);
      pipe_barrier(PIPE_V);
      RoundMode v432 = RoundMode::CAST_FLOOR;
      SaturationMode v433 = SaturationMode::ON;
      TCVT(v430, v428, v432, v433);
      // pto: %ri1_high_inline232__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v434 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %ri1_high_inline232__ssa_v0
      ;
      uint64_t v435 = (uint64_t) v53;
      TASSIGN(v434, v435);
      pipe_barrier(PIPE_V);
      RoundMode v436 = RoundMode::CAST_ROUND;
      SaturationMode v437 = SaturationMode::ON;
      TCVT(v434, v430, v436, v437);
      // pto: %t__tmp_v20
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v438 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v20
      ;
      uint64_t v439 = (uint64_t) v53;
      TASSIGN(v438, v439);
      pipe_barrier(PIPE_V);
      TMULS(v438, v434, v93);
      // pto: %ri1_bit_inline266__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v440 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %ri1_bit_inline266__ssa_v0
      ;
      uint64_t v441 = (uint64_t) v52;
      TASSIGN(v440, v441);
      pipe_barrier(PIPE_V);
      TSUB(v440, v424, v438);
      // pto: %t__tmp_v21
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v442 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v21
      ;
      uint64_t v443 = (uint64_t) v52;
      TASSIGN(v442, v443);
      pipe_barrier(PIPE_V);
      TMULS(v442, v440, v93);
      // pto: %t__tmp_v22
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v444 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v22
      ;
      uint64_t v445 = (uint64_t) v52;
      TASSIGN(v444, v445);
      pipe_barrier(PIPE_V);
      TSUB(v444, v414, v442);
      // pto: %t__tmp_v23
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v446 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v23
      ;
      uint64_t v447 = (uint64_t) v52;
      TASSIGN(v446, v447);
      pipe_barrier(PIPE_V);
      TADDS(v446, v444, v91);
      // pto: %ri1_inline230__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v448 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %ri1_inline230__ssa_v1
      ;
      uint64_t v449 = (uint64_t) v52;
      TASSIGN(v448, v449);
      pipe_barrier(PIPE_V);
      RoundMode v450 = RoundMode::CAST_ROUND;
      SaturationMode v451 = SaturationMode::ON;
      TCVT(v448, v446, v450, v451);
      // pto: %t__tmp_v24
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v452 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v24
      ;
      uint64_t v453 = (uint64_t) v53;
      TASSIGN(v452, v453);
      TMULS(v452, v414, v92);
      // pto: %t__tmp_v25
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v454 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v25
      ;
      uint64_t v455 = (uint64_t) v53;
      TASSIGN(v454, v455);
      pipe_barrier(PIPE_V);
      RoundMode v456 = RoundMode::CAST_FLOOR;
      SaturationMode v457 = SaturationMode::ON;
      TCVT(v454, v452, v456, v457);
      // pto: %ri2_low_inline267__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v458 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %ri2_low_inline267__ssa_v0
      ;
      uint64_t v459 = (uint64_t) v53;
      TASSIGN(v458, v459);
      pipe_barrier(PIPE_V);
      RoundMode v460 = RoundMode::CAST_ROUND;
      SaturationMode v461 = SaturationMode::ON;
      TCVT(v458, v454, v460, v461);
      // pto: %t__tmp_v26
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v462 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v26
      ;
      uint64_t v463 = (uint64_t) v48;
      TASSIGN(v462, v463);
      TMULS(v462, v414, v94);
      // pto: %t__tmp_v27
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v464 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v27
      ;
      uint64_t v465 = (uint64_t) v48;
      TASSIGN(v464, v465);
      pipe_barrier(PIPE_V);
      RoundMode v466 = RoundMode::CAST_FLOOR;
      SaturationMode v467 = SaturationMode::ON;
      TCVT(v464, v462, v466, v467);
      // pto: %ri2_high_inline162__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v468 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %ri2_high_inline162__ssa_v0
      ;
      uint64_t v469 = (uint64_t) v48;
      TASSIGN(v468, v469);
      pipe_barrier(PIPE_V);
      RoundMode v470 = RoundMode::CAST_ROUND;
      SaturationMode v471 = SaturationMode::ON;
      TCVT(v468, v464, v470, v471);
      // pto: %t__tmp_v28
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v472 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v28
      ;
      uint64_t v473 = (uint64_t) v48;
      TASSIGN(v472, v473);
      pipe_barrier(PIPE_V);
      TMULS(v472, v468, v93);
      // pto: %ri2_bit_inline281__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v474 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %ri2_bit_inline281__ssa_v0
      ;
      uint64_t v475 = (uint64_t) v53;
      TASSIGN(v474, v475);
      pipe_barrier(PIPE_V);
      TSUB(v474, v458, v472);
      // pto: %t__tmp_v29
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v476 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v29
      ;
      uint64_t v477 = (uint64_t) v53;
      TASSIGN(v476, v477);
      pipe_barrier(PIPE_V);
      TMULS(v476, v474, v95);
      // pto: %t__tmp_v30
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v478 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v30
      ;
      uint64_t v479 = (uint64_t) v53;
      TASSIGN(v478, v479);
      pipe_barrier(PIPE_V);
      TSUB(v478, v414, v476);
      // pto: %t__tmp_v31
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v480 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v31
      ;
      uint64_t v481 = (uint64_t) v53;
      TASSIGN(v480, v481);
      pipe_barrier(PIPE_V);
      TADDS(v480, v478, v93);
      // pto: %ri2_inline268__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v482 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %ri2_inline268__ssa_v0
      ;
      uint64_t v483 = (uint64_t) v53;
      TASSIGN(v482, v483);
      pipe_barrier(PIPE_V);
      RoundMode v484 = RoundMode::CAST_ROUND;
      SaturationMode v485 = SaturationMode::ON;
      TCVT(v482, v480, v484, v485);
      // pto: %t__tmp_v32
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v486 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v32
      ;
      uint64_t v487 = (uint64_t) v48;
      TASSIGN(v486, v487);
      TMULS(v486, v414, v94);
      // pto: %t__tmp_v33
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v488 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v33
      ;
      uint64_t v489 = (uint64_t) v48;
      TASSIGN(v488, v489);
      pipe_barrier(PIPE_V);
      RoundMode v490 = RoundMode::CAST_FLOOR;
      SaturationMode v491 = SaturationMode::ON;
      TCVT(v488, v486, v490, v491);
      // pto: %ci1_low_inline269__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v492 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %ci1_low_inline269__ssa_v0
      ;
      uint64_t v493 = (uint64_t) v48;
      TASSIGN(v492, v493);
      pipe_barrier(PIPE_V);
      RoundMode v494 = RoundMode::CAST_ROUND;
      SaturationMode v495 = SaturationMode::ON;
      TCVT(v492, v488, v494, v495);
      // pto: %t__tmp_v34
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v496 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v34
      ;
      uint64_t v497 = (uint64_t) v54;
      TASSIGN(v496, v497);
      TMULS(v496, v414, v96);
      // pto: %t__tmp_v35
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v498 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v35
      ;
      uint64_t v499 = (uint64_t) v54;
      TASSIGN(v498, v499);
      pipe_barrier(PIPE_V);
      RoundMode v500 = RoundMode::CAST_FLOOR;
      SaturationMode v501 = SaturationMode::ON;
      TCVT(v498, v496, v500, v501);
      // pto: %ci1_high_inline270__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v502 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %ci1_high_inline270__ssa_v0
      ;
      uint64_t v503 = (uint64_t) v54;
      TASSIGN(v502, v503);
      pipe_barrier(PIPE_V);
      RoundMode v504 = RoundMode::CAST_ROUND;
      SaturationMode v505 = SaturationMode::ON;
      TCVT(v502, v498, v504, v505);
      // pto: %t__tmp_v36
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v506 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v36
      ;
      uint64_t v507 = (uint64_t) v54;
      TASSIGN(v506, v507);
      pipe_barrier(PIPE_V);
      TMULS(v506, v502, v93);
      // pto: %ci1_bit_inline272__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v508 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %ci1_bit_inline272__ssa_v0
      ;
      uint64_t v509 = (uint64_t) v48;
      TASSIGN(v508, v509);
      pipe_barrier(PIPE_V);
      TSUB(v508, v492, v506);
      // pto: %t__tmp_v37
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v510 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v37
      ;
      uint64_t v511 = (uint64_t) v48;
      TASSIGN(v510, v511);
      pipe_barrier(PIPE_V);
      TMULS(v510, v508, v97);
      // pto: %t__tmp_v38
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v512 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v38
      ;
      uint64_t v513 = (uint64_t) v48;
      TASSIGN(v512, v513);
      pipe_barrier(PIPE_V);
      TSUB(v512, v414, v510);
      // pto: %t__tmp_v39
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v514 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v39
      ;
      uint64_t v515 = (uint64_t) v48;
      TASSIGN(v514, v515);
      pipe_barrier(PIPE_V);
      TADDS(v514, v512, v95);
      // pto: %ci1_inline275__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v516 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %ci1_inline275__ssa_v0
      ;
      uint64_t v517 = (uint64_t) v48;
      TASSIGN(v516, v517);
      pipe_barrier(PIPE_V);
      RoundMode v518 = RoundMode::CAST_ROUND;
      SaturationMode v519 = SaturationMode::ON;
      TCVT(v516, v514, v518, v519);
      // pto: %t__tmp_v40
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v520 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v40
      ;
      uint64_t v521 = (uint64_t) v54;
      TASSIGN(v520, v521);
      TMULS(v520, v414, v96);
      // pto: %t__tmp_v41
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v522 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v41
      ;
      uint64_t v523 = (uint64_t) v54;
      TASSIGN(v522, v523);
      pipe_barrier(PIPE_V);
      RoundMode v524 = RoundMode::CAST_FLOOR;
      SaturationMode v525 = SaturationMode::ON;
      TCVT(v522, v520, v524, v525);
      // pto: %ci2_low_inline276__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v526 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %ci2_low_inline276__ssa_v0
      ;
      uint64_t v527 = (uint64_t) v54;
      TASSIGN(v526, v527);
      pipe_barrier(PIPE_V);
      RoundMode v528 = RoundMode::CAST_ROUND;
      SaturationMode v529 = SaturationMode::ON;
      TCVT(v526, v522, v528, v529);
      // pto: %t__tmp_v42
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v530 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v42
      ;
      uint64_t v531 = (uint64_t) v50;
      TASSIGN(v530, v531);
      TMULS(v530, v414, v98);
      // pto: %t__tmp_v43
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v532 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v43
      ;
      uint64_t v533 = (uint64_t) v50;
      TASSIGN(v532, v533);
      pipe_barrier(PIPE_V);
      RoundMode v534 = RoundMode::CAST_FLOOR;
      SaturationMode v535 = SaturationMode::ON;
      TCVT(v532, v530, v534, v535);
      // pto: %ci2_high_inline146__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v536 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %ci2_high_inline146__ssa_v0
      ;
      uint64_t v537 = (uint64_t) v50;
      TASSIGN(v536, v537);
      pipe_barrier(PIPE_V);
      RoundMode v538 = RoundMode::CAST_ROUND;
      SaturationMode v539 = SaturationMode::ON;
      TCVT(v536, v532, v538, v539);
      // pto: %t__tmp_v44
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v540 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v44
      ;
      uint64_t v541 = (uint64_t) v50;
      TASSIGN(v540, v541);
      pipe_barrier(PIPE_V);
      TMULS(v540, v536, v93);
      // pto: %ci2_bit_inline277__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v542 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %ci2_bit_inline277__ssa_v0
      ;
      uint64_t v543 = (uint64_t) v54;
      TASSIGN(v542, v543);
      pipe_barrier(PIPE_V);
      TSUB(v542, v526, v540);
      // pto: %t__tmp_v45
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v544 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v45
      ;
      uint64_t v545 = (uint64_t) v54;
      TASSIGN(v544, v545);
      pipe_barrier(PIPE_V);
      TMULS(v544, v542, v99);
      // pto: %t__tmp_v46
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v546 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v46
      ;
      uint64_t v547 = (uint64_t) v51;
      TASSIGN(v546, v547);
      pipe_barrier(PIPE_V);
      TSUB(v546, v414, v544);
      // pto: %t__tmp_v47
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v548 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v47
      ;
      uint64_t v549 = (uint64_t) v51;
      TASSIGN(v548, v549);
      pipe_barrier(PIPE_V);
      TADDS(v548, v546, v97);
      // pto: %ci2_inline238__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v550 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %ci2_inline238__ssa_v0
      ;
      uint64_t v551 = (uint64_t) v51;
      TASSIGN(v550, v551);
      pipe_barrier(PIPE_V);
      RoundMode v552 = RoundMode::CAST_ROUND;
      SaturationMode v553 = SaturationMode::ON;
      TCVT(v550, v548, v552, v553);
      // pto: %t__tmp_v48
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v554 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v48
      ;
      uint64_t v555 = (uint64_t) v54;
      TASSIGN(v554, v555);
      TGATHER(v554, v404, v448, v412);
      // pto: %max2_inline121__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v556 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %max2_inline121__ssa_v1
      ;
      uint64_t v557 = (uint64_t) v54;
      TASSIGN(v556, v557);
      pipe_barrier(PIPE_V);
      TMAX(v556, v404, v554);
      // pto: %t__tmp_v49
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v558 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v49
      ;
      uint64_t v559 = (uint64_t) v50;
      TASSIGN(v558, v559);
      pipe_barrier(PIPE_V);
      TGATHER(v558, v556, v482, v412);
      // pto: %max4_inline242__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v560 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %max4_inline242__ssa_v1
      ;
      uint64_t v561 = (uint64_t) v54;
      TASSIGN(v560, v561);
      pipe_barrier(PIPE_V);
      TMAX(v560, v556, v558);
      // pto: %t__tmp_v50
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v562 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v50
      ;
      uint64_t v563 = (uint64_t) v54;
      TASSIGN(v562, v563);
      pipe_barrier(PIPE_V);
      TSUB(v562, v404, v560);
      // pto: %exps_inline243__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v564 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %exps_inline243__ssa_v1
      ;
      uint64_t v565 = (uint64_t) v54;
      TASSIGN(v564, v565);
      pipe_barrier(PIPE_V);
      TEXP(v564, v562);
      // pto: %t__tmp_v51
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v566 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v51
      ;
      uint64_t v567 = (uint64_t) v49;
      TASSIGN(v566, v567);
      pipe_barrier(PIPE_V);
      TGATHER(v566, v564, v448, v412);
      // pto: %sum2_inline246__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v568 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %sum2_inline246__ssa_v1
      ;
      uint64_t v569 = (uint64_t) v52;
      TASSIGN(v568, v569);
      pipe_barrier(PIPE_V);
      TADD(v568, v564, v566);
      // pto: %t__tmp_v52
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v570 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v52
      ;
      uint64_t v571 = (uint64_t) v49;
      TASSIGN(v570, v571);
      pipe_barrier(PIPE_V);
      TGATHER(v570, v568, v482, v412);
      // pto: %sum4_inline250__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v572 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %sum4_inline250__ssa_v1
      ;
      uint64_t v573 = (uint64_t) v52;
      TASSIGN(v572, v573);
      pipe_barrier(PIPE_V);
      TADD(v572, v568, v570);
      // pto: %t__tmp_v53
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v574 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v53
      ;
      uint64_t v575 = (uint64_t) v52;
      TASSIGN(v574, v575);
      pipe_barrier(PIPE_V);
      TDIV(v574, v564, v572);
      // pto: %probs_inline251__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v576 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %probs_inline251__ssa_v1
      ;
      uint64_t v577 = (uint64_t) v52;
      TASSIGN(v576, v577);
      pipe_barrier(PIPE_V);
      TADDS(v576, v574, v89);
      // pto: %t__tmp_v54
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v578 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v54
      ;
      uint64_t v579 = (uint64_t) v53;
      TASSIGN(v578, v579);
      pipe_barrier(PIPE_V);
      TGATHER(v578, v576, v516, v412);
      // pto: %col2_inline252__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v580 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %col2_inline252__ssa_v4
      ;
      uint64_t v581 = (uint64_t) v53;
      TASSIGN(v580, v581);
      pipe_barrier(PIPE_V);
      TADD(v580, v576, v578);
      // pto: %t__tmp_v55
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v582 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v55
      ;
      uint64_t v583 = (uint64_t) v48;
      TASSIGN(v582, v583);
      pipe_barrier(PIPE_V);
      TGATHER(v582, v580, v550, v412);
      // pto: %t__tmp_v56
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v584 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v56
      ;
      uint64_t v585 = (uint64_t) v40;
      TASSIGN(v584, v585);
      pipe_barrier(PIPE_V);
      TADD(v584, v580, v582);
      // pto: %col4_inline256__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v586 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %col4_inline256__ssa_v4
      ;
      uint64_t v587 = (uint64_t) v40;
      TASSIGN(v586, v587);
      pipe_barrier(PIPE_V);
      TADDS(v586, v584, v89);
      // pto: %cur_inline257__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v588 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %cur_inline257__ssa_v4
      ;
      uint64_t v589 = (uint64_t) v40;
      TASSIGN(v588, v589);
      pipe_barrier(PIPE_V);
      TDIV(v588, v576, v586);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      for (int64_t v590 = v84; v590 < v329; v590 += v71) {
        // pto: %119
        ;
        int64_t v591 = (int64_t) ((uint64_t) v590 * (uint64_t) v76);
        // pto: %sv00_inline280__ssa_v0
        ;
        float v592 = v588.GetValue(v591);
        // pto: %121
        ;
        int64_t v593 = (int64_t) ((uint64_t) v591 + (uint64_t) v71);
        // pto: %sv01_inline221__ssa_v0
        ;
        float v594 = v588.GetValue(v593);
        // pto: %125
        ;
        int64_t v595 = (int64_t) ((uint64_t) v591 + (uint64_t) v83);
        // pto: %sv02_inline285__ssa_v0
        ;
        float v596 = v588.GetValue(v595);
        // pto: %129
        ;
        int64_t v597 = (int64_t) ((uint64_t) v591 + (uint64_t) v74);
        // pto: %sv03_inline288__ssa_v0
        ;
        float v598 = v588.GetValue(v597);
        // pto: %133
        ;
        int64_t v599 = (int64_t) ((uint64_t) v591 + (uint64_t) v100);
        // pto: %sv04_inline158__ssa_v0
        ;
        float v600 = v588.GetValue(v599);
        // pto: %137
        ;
        int64_t v601 = (int64_t) ((uint64_t) v591 + (uint64_t) v101);
        // pto: %sv05_inline290__ssa_v0
        ;
        float v602 = v588.GetValue(v601);
        // pto: %141
        ;
        int64_t v603 = (int64_t) ((uint64_t) v591 + (uint64_t) v102);
        // pto: %sv06_inline163__ssa_v0
        ;
        float v604 = v588.GetValue(v603);
        // pto: %145
        ;
        int64_t v605 = (int64_t) ((uint64_t) v591 + (uint64_t) v103);
        // pto: %sv07_inline139__ssa_v0
        ;
        float v606 = v588.GetValue(v605);
        // pto: %149
        ;
        int64_t v607 = (int64_t) ((uint64_t) v591 + (uint64_t) v87);
        // pto: %sv08_inline292__ssa_v0
        ;
        float v608 = v588.GetValue(v607);
        // pto: %153
        ;
        int64_t v609 = (int64_t) ((uint64_t) v591 + (uint64_t) v104);
        // pto: %sv09_inline109__ssa_v0
        ;
        float v610 = v588.GetValue(v609);
        // pto: %157
        ;
        int64_t v611 = (int64_t) ((uint64_t) v591 + (uint64_t) v105);
        // pto: %sv10_inline104__ssa_v0
        ;
        float v612 = v588.GetValue(v611);
        // pto: %161
        ;
        int64_t v613 = (int64_t) ((uint64_t) v591 + (uint64_t) v75);
        // pto: %sv11_inline137__ssa_v0
        ;
        float v614 = v588.GetValue(v613);
        // pto: %165
        ;
        int64_t v615 = (int64_t) ((uint64_t) v591 + (uint64_t) v106);
        // pto: %sv12_inline298__ssa_v0
        ;
        float v616 = v588.GetValue(v615);
        // pto: %169
        ;
        int64_t v617 = (int64_t) ((uint64_t) v591 + (uint64_t) v107);
        // pto: %sv13_inline78__ssa_v0
        ;
        float v618 = v588.GetValue(v617);
        // pto: %173
        ;
        int64_t v619 = (int64_t) ((uint64_t) v591 + (uint64_t) v108);
        // pto: %sv14_inline293__ssa_v0
        ;
        float v620 = v588.GetValue(v619);
        // pto: %177
        ;
        int64_t v621 = (int64_t) ((uint64_t) v591 + (uint64_t) v109);
        // pto: %sv15_inline145__ssa_v0
        ;
        float v622 = v588.GetValue(v621);
        // pto: %sv00_inline280__rv_v2
        ;
        float v623;
        // pto: %sv01_inline221__rv_v2
        ;
        float v624;
        // pto: %sv02_inline285__rv_v2
        ;
        float v625;
        // pto: %sv03_inline288__rv_v2
        ;
        float v626;
        // pto: %sv04_inline158__rv_v2
        ;
        float v627;
        // pto: %sv05_inline290__rv_v2
        ;
        float v628;
        // pto: %sv06_inline163__rv_v2
        ;
        float v629;
        // pto: %sv07_inline139__rv_v2
        ;
        float v630;
        // pto: %sv08_inline292__rv_v2
        ;
        float v631;
        // pto: %sv09_inline109__rv_v2
        ;
        float v632;
        // pto: %sv10_inline104__rv_v2
        ;
        float v633;
        // pto: %sv11_inline137__rv_v2
        ;
        float v634;
        // pto: %sv12_inline298__rv_v2
        ;
        float v635;
        // pto: %sv13_inline78__rv_v2
        ;
        float v636;
        // pto: %sv14_inline293__rv_v2
        ;
        float v637;
        // pto: %sv15_inline145__rv_v2
        ;
        float v638;
        v623 = v592;
        v624 = v594;
        v625 = v596;
        v626 = v598;
        v627 = v600;
        v628 = v602;
        v629 = v604;
        v630 = v606;
        v631 = v608;
        v632 = v610;
        v633 = v612;
        v634 = v614;
        v635 = v616;
        v636 = v618;
        v637 = v620;
        v638 = v622;
        for (int64_t v639 = v84; v639 < v110; v639 += v71) {
          // pto: %180
          ;
          float v640 = v623 + v624;
          // pto: %181
          ;
          float v641 = v625 + v626;
          // pto: %182, %183, %184
          ;
          float v642 = v91 / (v640 + v641 + v89);
          // pto: %185
          ;
          float v643 = v623 * v642;
          // pto: %186
          ;
          float v644 = v624 * v642;
          // pto: %187
          ;
          float v645 = v625 * v642;
          // pto: %188
          ;
          float v646 = v626 * v642;
          // pto: %189
          ;
          float v647 = v627 + v628;
          // pto: %190
          ;
          float v648 = v629 + v630;
          // pto: %191, %192, %193
          ;
          float v649 = v91 / (v647 + v648 + v89);
          // pto: %194
          ;
          float v650 = v627 * v649;
          // pto: %195
          ;
          float v651 = v628 * v649;
          // pto: %196
          ;
          float v652 = v629 * v649;
          // pto: %197
          ;
          float v653 = v630 * v649;
          // pto: %198
          ;
          float v654 = v631 + v632;
          // pto: %199
          ;
          float v655 = v633 + v634;
          // pto: %200, %201, %202
          ;
          float v656 = v91 / (v654 + v655 + v89);
          // pto: %203
          ;
          float v657 = v631 * v656;
          // pto: %204
          ;
          float v658 = v632 * v656;
          // pto: %205
          ;
          float v659 = v633 * v656;
          // pto: %206
          ;
          float v660 = v634 * v656;
          // pto: %207
          ;
          float v661 = v635 + v636;
          // pto: %208
          ;
          float v662 = v637 + v638;
          // pto: %209, %210, %211
          ;
          float v663 = v91 / (v661 + v662 + v89);
          // pto: %212
          ;
          float v664 = v635 * v663;
          // pto: %213
          ;
          float v665 = v636 * v663;
          // pto: %214
          ;
          float v666 = v637 * v663;
          // pto: %215
          ;
          float v667 = v638 * v663;
          // pto: %216, %217, %218, %219, %220
          ;
          float v668 = v91 / ((v643 + v650) + (v657 + v664) + v89);
          // pto: %221
          ;
          // pto: %222
          ;
          // pto: %223
          ;
          // pto: %224
          ;
          // pto: %225, %226, %227, %228, %229
          ;
          float v669 = v91 / ((v644 + v651) + (v658 + v665) + v89);
          // pto: %230
          ;
          // pto: %231
          ;
          // pto: %232
          ;
          // pto: %233
          ;
          // pto: %234, %235, %236, %237, %238
          ;
          float v670 = v91 / ((v645 + v652) + (v659 + v666) + v89);
          // pto: %239
          ;
          // pto: %240
          ;
          // pto: %241
          ;
          // pto: %242
          ;
          // pto: %243, %244, %245, %246, %247
          ;
          float v671 = v91 / ((v646 + v653) + (v660 + v667) + v89);
          // pto: %248
          ;
          // pto: %249
          ;
          // pto: %250
          ;
          // pto: %251
          ;
          v623 = v643 * v668;
          v624 = v644 * v669;
          v625 = v645 * v670;
          v626 = v646 * v671;
          v627 = v650 * v668;
          v628 = v651 * v669;
          v629 = v652 * v670;
          v630 = v653 * v671;
          v631 = v657 * v668;
          v632 = v658 * v669;
          v633 = v659 * v670;
          v634 = v660 * v671;
          v635 = v664 * v668;
          v636 = v665 * v669;
          v637 = v666 * v670;
          v638 = v667 * v671;
        };
        v588.SetValue(v591, v623);
        v588.SetValue(v593, v624);
        v588.SetValue(v595, v625);
        v588.SetValue(v597, v626);
        v588.SetValue(v599, v627);
        v588.SetValue(v601, v628);
        v588.SetValue(v603, v629);
        v588.SetValue(v605, v630);
        v588.SetValue(v607, v631);
        v588.SetValue(v609, v632);
        v588.SetValue(v611, v633);
        v588.SetValue(v613, v634);
        v588.SetValue(v615, v635);
        v588.SetValue(v617, v636);
        v588.SetValue(v619, v637);
        v588.SetValue(v621, v638);
      };
      set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
      // pto: %t__tmp_v57
      ;
      Tile<TileType::Vec, float, 2, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v672 = Tile<TileType::Vec, float, 2, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v83, v76);
      // pto: %t__tmp_v57
      ;
      uint64_t v673 = (uint64_t) v40;
      TASSIGN(v672, v673);
      v672.SetValidShape(v329, v76);
      // pto: %comb_inline153__ssa_v0_pview
      ;
      const int64_t v674 = 0;
      // pto: %comb_inline153__ssa_v0_pview
      ;
      __gm__ float* v675 = PTOAS__GLOBAL_TENSOR_DATA(v166);
      // pto: %comb_inline153__ssa_v0_pview
      ;
      const int64_t v676 = 1;
      // pto: %comb_inline153__ssa_v0_pview
      ;
      const int64_t v677 = 1;
      // pto: %comb_inline153__ssa_v0_pview
      ;
      const int64_t v678 = 1;
      // pto: %comb_inline153__ssa_v0_pview
      ;
      int64_t v679 = v329 * v76;
      // pto: %comb_inline153__ssa_v0_pview
      ;
      int64_t v680 = v678 * v679;
      // pto: %comb_inline153__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, 16> v681 = pto::Shape<1, 1, 1, -1, 16>(v676, v677, v678, v329, v76);
      // pto: %comb_inline153__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v682 = pto::Stride<-1, -1, -1, -1, -1>(v677 * v680, v680, v679, v76, v71);
      // pto: %comb_inline153__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v683 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v675 + (v674 + v353 * v76 + v84 * v71), v681, v682);
      wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      TSTORE(v683, v672);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    };
  } else {
    // pto: %316
    ;
    int64_t v684 = (int64_t) ((uint64_t) v319 - (uint64_t) v18);
    // pto: %318
    ;
    int64_t v685 = v684 < v22 ? v684 : v22;
    // pto: %317, %319, %320
    ;
    int64_t v686 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v684 * (uint64_t) v21) + (uint64_t) v685) * (uint64_t) v83);
    // pto: %321
    ;
    int64_t v687 = (int64_t) ((uint64_t) v22 - (uint64_t) v684);
    // pto: %322
    ;
    int64_t v688 = v687 < v84 ? v84 : v687;
    // pto: %323
    ;
    int64_t v689 = v688 < v71 ? v688 : v71;
    // pto: %326, %324, %325
    ;
    int64_t v690 = (int64_t) ((uint64_t) v686 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v21 + (uint64_t) v689) * (uint64_t) v83));
    // pto: %327
    ;
    // pto: %328
    ;
    // pto: %d_begin_inline297__phi_v2
    ;
    int64_t v691;
    // pto: %d_end_inline114__phi_v2
    ;
    int64_t v692;
    // pto: %work_begin_inline41__phi_v3
    ;
    int64_t v693;
    // pto: %work_end_inline38__phi_v3
    ;
    int64_t v694;
    if (v24 < v25) {
      // pto: %329
      ;
      // pto: %d_worker_inline71__phi_v2
      ;
      int64_t v695;
      // pto: %d_workers_inline244__phi_v2
      ;
      int64_t v696;
      // pto: %pair_id_inline271__phi_v2
      ;
      int64_t v697;
      if (v684 < v26) {
        // pto: %330
        ;
        int64_t v698 = (int64_t) ((uint64_t) v27 + (uint64_t) v71);
        // pto: %331
        ;
        // pto: %333
        ;
        v695 = v684 % v698;
        v696 = v698;
        v697 = v684 / v698;
      } else {
        // pto: %335
        ;
        int64_t v699 = (int64_t) ((uint64_t) v684 - (uint64_t) v26);
        // pto: %337, %336
        ;
        // pto: %338
        ;
        v695 = v699 % v27;
        v696 = v27;
        v697 = (int64_t) ((uint64_t) v28 + (uint64_t) (v699 / v27));
      };
      // pto: %339
      ;
      uint64_t v700 = (uint64_t) v697;
      // pto: %339
      ;
      int64_t v701 = (int64_t) (v700 * (uint64_t) v83);
      // pto: %340
      ;
      int64_t v702 = v23 < v701 ? v23 : v701;
      // pto: %341
      ;
      int64_t v703 = (int64_t) ((uint64_t) v702 + (uint64_t) v83);
      // pto: %342
      ;
      // pto: %343
      ;
      int64_t v704 = v87 / v696;
      // pto: %344
      ;
      int64_t v705 = v87 % v696;
      // pto: %345
      ;
      uint64_t v706 = (uint64_t) v695;
      // pto: %346
      ;
      bool v707 = v695 < v705;
      // pto: %346
      ;
      int64_t v708 = v707 ? v695 : v705;
      // pto: %345, %347
      ;
      int64_t v709 = (int64_t) ((uint64_t) ((int64_t) v706 * (uint64_t) v704) + (uint64_t) v708);
      // pto: %349
      ;
      uint64_t v710 = (uint64_t) v695;
      // pto: %349
      ;
      int64_t v711 = (int64_t) ((uint64_t) v705 - v710);
      // pto: %350
      ;
      int64_t v712 = v711 < v84 ? v84 : v711;
      // pto: %351
      ;
      int64_t v713 = v712 < v71 ? v712 : v71;
      // pto: %348, %352
      ;
      v691 = v709;
      v692 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v709 + (uint64_t) v704) + (uint64_t) v713);
      v693 = v702;
      v694 = v23 < v703 ? v23 : v703;
    } else {
      v691 = v84;
      v692 = v87;
      v693 = v686;
      v694 = v23 < v690 ? v23 : v690;
    };
    // pto: %353
    ;
    bool v714 = v25 <= v684;
    // pto: %work_begin_inline41__phi_v5
    ;
    int64_t v715 = v714 ? v84 : v693;
    // pto: %work_end_inline38__phi_v5
    ;
    int64_t v716 = v714 ? v84 : v694;
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    for (int64_t v717 = v715; v717 < v716; v717 += v83) {
      // pto: %354
      ;
      int64_t v718 = (int64_t) ((uint64_t) v716 - (uint64_t) v717);
      // pto: %355
      ;
      int64_t v719 = v718 < v83 ? v718 : v83;
      // pto: %356
      ;
      int64_t v720 = (int64_t) ((uint64_t) v719 * (uint64_t) v70);
      // pto: %m_parts_load_inline302__ssa_v0
      ;
      Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v721 = Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v75, v720);
      // pto: %m_parts_load_inline302__ssa_v0
      ;
      uint64_t v722 = (uint64_t) v40;
      TASSIGN(v721, v722);
      // pto: %357
      ;
      int64_t v723 = (int64_t) ((uint64_t) v717 * (uint64_t) v70);
      // pto: %358
      ;
      int64_t v724 = v723 < v84 ? v84 : v723;
      // pto: %360
      ;
      const int64_t v725 = 0;
      // pto: %360
      ;
      __gm__ float* v726 = PTOAS__GLOBAL_TENSOR_DATA(v140);
      // pto: %360
      ;
      const int64_t v727 = 1;
      // pto: %360
      ;
      const int64_t v728 = 1;
      // pto: %360
      ;
      const int64_t v729 = 1;
      // pto: %360
      ;
      int64_t v730 = v75 * v132;
      // pto: %360
      ;
      int64_t v731 = v729 * v730;
      // pto: %360
      ;
      pto::Shape<1, 1, 1, 11, -1> v732 = pto::Shape<1, 1, 1, 11, -1>(v727, v728, v729, v75, v720);
      // pto: %360
      ;
      pto::Stride<-1, -1, -1, -1, -1> v733 = pto::Stride<-1, -1, -1, -1, -1>(v728 * v731, v731, v730, v132, v71);
      // pto: %360
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 11, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v734 = GlobalTensor<float, pto::Shape<1, 1, 1, 11, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v726 + (v725 + v84 * v132 + v724 * v71), v732, v733);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      TLOAD(v721, v734);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      // pto: %t__tmp_v58
      ;
      Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v735 = Tile<TileType::Vec, float, 11, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v75, v86);
      // pto: %t__tmp_v58
      ;
      uint64_t v736 = (uint64_t) v40;
      TASSIGN(v735, v736);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      TFILLPAD<pto::TFillPadMode::InPlace>(v735, v721);
      v735.SetValidShape(v75, v86);
      // pto: %t__tmp_v59
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v737 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v86);
      // pto: %t__tmp_v59
      ;
      uint64_t v738 = (uint64_t) v40;
      TASSIGN(v737, v738);
      pipe_barrier(PIPE_V);
      TCOLSUM(v737, v735);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      // pto: %m_total_inline310__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v739 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v70);
      // pto: %m_total_inline310__ssa_v0
      ;
      uint64_t v740 = (uint64_t) v40;
      TASSIGN(v739, v740);
      // pto: %m_sq_load_inline303__ssa_v0
      ;
      Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v741 = Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v75, v719);
      // pto: %m_sq_load_inline303__ssa_v0
      ;
      uint64_t v742 = (uint64_t) v44;
      TASSIGN(v741, v742);
      // pto: %361
      ;
      int64_t v743 = v717 < v84 ? v84 : v717;
      // pto: %362
      ;
      int64_t v744 = (int64_t) v29;
      // pto: %362
      ;
      const int64_t v745 = 0;
      // pto: %362
      ;
      __gm__ float* v746 = PTOAS__GLOBAL_TENSOR_DATA(v149);
      // pto: %362
      ;
      const int64_t v747 = 1;
      // pto: %362
      ;
      const int64_t v748 = 1;
      // pto: %362
      ;
      const int64_t v749 = 1;
      // pto: %362
      ;
      int64_t v750 = v75 * v744;
      // pto: %362
      ;
      int64_t v751 = v749 * v750;
      // pto: %362
      ;
      pto::Shape<1, 1, 1, 11, -1> v752 = pto::Shape<1, 1, 1, 11, -1>(v747, v748, v749, v75, v719);
      // pto: %362
      ;
      pto::Stride<-1, -1, -1, -1, -1> v753 = pto::Stride<-1, -1, -1, -1, -1>(v748 * v751, v751, v750, v744, v71);
      // pto: %362
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 11, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v754 = GlobalTensor<float, pto::Shape<1, 1, 1, 11, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v746 + (v745 + v84 * v744 + v743 * v71), v752, v753);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      TLOAD(v741, v754);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
      // pto: %t__tmp_v60
      ;
      Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v755 = Tile<TileType::Vec, float, 11, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v75, v87);
      // pto: %t__tmp_v60
      ;
      uint64_t v756 = (uint64_t) v44;
      TASSIGN(v755, v756);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
      TFILLPAD<pto::TFillPadMode::InPlace>(v755, v741);
      v755.SetValidShape(v75, v87);
      // pto: %m_rms_tmp_inline306__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v757 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %m_rms_tmp_inline306__ssa_v0
      ;
      uint64_t v758 = (uint64_t) v45;
      TASSIGN(v757, v758);
      // pto: %t__tmp_v61
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v759 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %t__tmp_v61
      ;
      uint64_t v760 = (uint64_t) v44;
      TASSIGN(v759, v760);
      pipe_barrier(PIPE_V);
      TCOLSUM(v759, v755);
      // pto: %t__tmp_v62
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v761 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %t__tmp_v62
      ;
      uint64_t v762 = (uint64_t) v44;
      TASSIGN(v761, v762);
      pipe_barrier(PIPE_V);
      TMULS(v761, v759, v88);
      // pto: %t__tmp_v63
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v763 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %t__tmp_v63
      ;
      uint64_t v764 = (uint64_t) v44;
      TASSIGN(v763, v764);
      pipe_barrier(PIPE_V);
      TADDS(v763, v761, v89);
      // pto: %t__tmp_v64
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v765 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %t__tmp_v64
      ;
      uint64_t v766 = (uint64_t) v55;
      TASSIGN(v765, v766);
      pipe_barrier(PIPE_V);
      TRSQRT(v765, v763, v757);
      // pto: %m_inv_inline215__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v767 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v71);
      // pto: %m_inv_inline215__ssa_v0
      ;
      uint64_t v768 = (uint64_t) v55;
      TASSIGN(v767, v768);
      // pto: %gate_base_inline308__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v769 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %gate_base_inline308__ssa_v0
      ;
      uint64_t v770 = (uint64_t) v56;
      TASSIGN(v769, v770);
      // pto: %363
      ;
      __gm__ float* v771 = PTOAS__GLOBAL_TENSOR_DATA(v157);
      // pto: %363
      ;
      pto::Shape<1, 1, 1, 1, 8> v772 = pto::Shape<1, 1, 1, 1, 8>();
      // pto: %363
      ;
      pto::Stride<24, 24, 24, 24, 1> v773 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %363
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v774 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v771, v772, v773);
      TLOAD(v769, v774);
      // pto: %gate_scales_inline144__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v775 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v74);
      // pto: %gate_scales_inline144__ssa_v0
      ;
      uint64_t v776 = (uint64_t) v57;
      TASSIGN(v775, v776);
      // pto: %hc_scale_2d_inline132__ssa_v0_pview
      ;
      __gm__ float* v777 = PTOAS__GLOBAL_TENSOR_DATA(v174);
      // pto: %hc_scale_2d_inline132__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 3> v778 = pto::Shape<1, 1, 1, 1, 3>();
      // pto: %hc_scale_2d_inline132__ssa_v0_pview
      ;
      pto::Stride<3, 3, 3, 3, 1> v779 = pto::Stride<3, 3, 3, 3, 1>();
      // pto: %hc_scale_2d_inline132__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND> v780 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND>(v777, v778, v779);
      TLOAD(v775, v780);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %t__ci_tmp_v1
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v781 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v90);
      // pto: %t__ci_tmp_v1
      ;
      uint64_t v782 = (uint64_t) v44;
      TASSIGN(v781, v782);
      // pto: %t__tmp_v65
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v783 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %t__tmp_v65
      ;
      uint64_t v784 = (uint64_t) v58;
      TASSIGN(v783, v784);
      // pto: %364
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v785;
      TRESHAPE(v785, v781);
      // pto: %365
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v786;
      TRESHAPE(v786, v783);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v786, v78, v785);
      // pto: %gate_ids_inline68__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v787 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %gate_ids_inline68__ssa_v0
      ;
      uint64_t v788 = (uint64_t) v44;
      TASSIGN(v787, v788);
      pipe_barrier(PIPE_V);
      RoundMode v789 = RoundMode::CAST_ROUND;
      SaturationMode v790 = SaturationMode::ON;
      TCVT(v787, v783, v789, v790);
      // pto: %t__tmp_v66
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v791 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %t__tmp_v66
      ;
      uint64_t v792 = (uint64_t) v44;
      TASSIGN(v791, v792);
      pipe_barrier(PIPE_V);
      TMULS(v791, v787, v94);
      // pto: %gate_scale_ids_inline309__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v793 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %gate_scale_ids_inline309__ssa_v0
      ;
      uint64_t v794 = (uint64_t) v44;
      TASSIGN(v793, v794);
      pipe_barrier(PIPE_V);
      RoundMode v795 = RoundMode::CAST_FLOOR;
      SaturationMode v796 = SaturationMode::ON;
      TCVT(v793, v791, v795, v796);
      // pto: %gate_tmp_inline184__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v797 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %gate_tmp_inline184__ssa_v0
      ;
      uint64_t v798 = (uint64_t) v59;
      TASSIGN(v797, v798);
      // pto: %gate_scale_vec_inline307__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v799 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %gate_scale_vec_inline307__ssa_v0
      ;
      uint64_t v800 = (uint64_t) v46;
      TASSIGN(v799, v800);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TGATHER(v799, v775, v793, v797);
      // pto: %t__tmp_v67
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v801 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v87);
      // pto: %t__tmp_v67
      ;
      uint64_t v802 = (uint64_t) v40;
      TASSIGN(v801, v802);
      // pto: %366
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v803;
      // pto: %366
      ;
      uint64_t v804 = (uint64_t) v40;
      TASSIGN(v803, v804);
      // pto: %gate_raw_inline311__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v805 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v87);
      // pto: %gate_raw_inline311__ssa_v0
      ;
      uint64_t v806 = (uint64_t) v40;
      TASSIGN(v805, v806);
      TADDS(v805, v803, v80);
      // pto: %gate_norm_inline49__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v807 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v87);
      // pto: %gate_norm_inline49__ssa_v0
      ;
      uint64_t v808 = (uint64_t) v40;
      TASSIGN(v807, v808);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v807, v805, v767);
      // pto: %gate_scaled_inline312__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v809 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v87);
      // pto: %gate_scaled_inline312__ssa_v0
      ;
      uint64_t v810 = (uint64_t) v40;
      TASSIGN(v809, v810);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v809, v807, v799);
      // pto: %t__tmp_v68
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v811 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v87);
      // pto: %t__tmp_v68
      ;
      uint64_t v812 = (uint64_t) v52;
      TASSIGN(v811, v812);
      TCOLEXPAND(v811, v769);
      // pto: %gate_logits_inline98__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v813 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v87);
      // pto: %gate_logits_inline98__ssa_v0
      ;
      uint64_t v814 = (uint64_t) v40;
      TASSIGN(v813, v814);
      pipe_barrier(PIPE_V);
      TADD(v813, v809, v811);
      // pto: %t__tmp_v69
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v815 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v87);
      // pto: %t__tmp_v69
      ;
      uint64_t v816 = (uint64_t) v40;
      TASSIGN(v815, v816);
      pipe_barrier(PIPE_V);
      TNEG(v815, v813);
      // pto: %t__tmp_v70
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v817 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v87);
      // pto: %t__tmp_v70
      ;
      uint64_t v818 = (uint64_t) v40;
      TASSIGN(v817, v818);
      pipe_barrier(PIPE_V);
      TEXP(v817, v815);
      // pto: %t__tmp_v71
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v819 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v87);
      // pto: %t__tmp_v71
      ;
      uint64_t v820 = (uint64_t) v40;
      TASSIGN(v819, v820);
      pipe_barrier(PIPE_V);
      TADDS(v819, v817, v91);
      // pto: %gate_sigmoid_inline148__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v821 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v87);
      // pto: %gate_sigmoid_inline148__ssa_v0
      ;
      uint64_t v822 = (uint64_t) v49;
      TASSIGN(v821, v822);
      pipe_barrier(PIPE_V);
      TRECIP(v821, v819);
      // pto: %transpose_tmp
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v823 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v87);
      // pto: %transpose_tmp
      ;
      uint64_t v824 = (uint64_t) v40;
      TASSIGN(v823, v824);
      // pto: %gate_transposed_inline123__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v825 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v87);
      // pto: %gate_transposed_inline123__ssa_v0
      ;
      uint64_t v826 = (uint64_t) v44;
      TASSIGN(v825, v826);
      pipe_barrier(PIPE_V);
      TTRANS(v825, v821, v823);
      // pto: %t__ci_tmp_v2
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v827 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v90);
      // pto: %t__ci_tmp_v2
      ;
      uint64_t v828 = (uint64_t) v40;
      TASSIGN(v827, v828);
      // pto: %t__tmp_v72
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v829 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v72
      ;
      uint64_t v830 = (uint64_t) v50;
      TASSIGN(v829, v830);
      // pto: %367
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v831;
      TRESHAPE(v831, v827);
      // pto: %368
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v832;
      TRESHAPE(v832, v829);
      TCI<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v832, v78, v831);
      // pto: %post_ids_float_inline313__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v833 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %post_ids_float_inline313__ssa_v0
      ;
      uint64_t v834 = (uint64_t) v40;
      TASSIGN(v833, v834);
      pipe_barrier(PIPE_V);
      RoundMode v835 = RoundMode::CAST_ROUND;
      SaturationMode v836 = SaturationMode::ON;
      TCVT(v833, v829, v835, v836);
      // pto: %t__tmp_v73
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v837 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v73
      ;
      uint64_t v838 = (uint64_t) v51;
      TASSIGN(v837, v838);
      pipe_barrier(PIPE_V);
      TMULS(v837, v833, v94);
      // pto: %t__tmp_v74
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v839 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v74
      ;
      uint64_t v840 = (uint64_t) v51;
      TASSIGN(v839, v840);
      pipe_barrier(PIPE_V);
      RoundMode v841 = RoundMode::CAST_FLOOR;
      SaturationMode v842 = SaturationMode::ON;
      TCVT(v839, v837, v841, v842);
      // pto: %post_rows_inline245__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v843 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %post_rows_inline245__ssa_v0
      ;
      uint64_t v844 = (uint64_t) v51;
      TASSIGN(v843, v844);
      pipe_barrier(PIPE_V);
      RoundMode v845 = RoundMode::CAST_ROUND;
      SaturationMode v846 = SaturationMode::ON;
      TCVT(v843, v839, v845, v846);
      // pto: %t__tmp_v75
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v847 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v75
      ;
      uint64_t v848 = (uint64_t) v51;
      TASSIGN(v847, v848);
      pipe_barrier(PIPE_V);
      TMULS(v847, v843, v95);
      // pto: %t__tmp_v76
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v849 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v76
      ;
      uint64_t v850 = (uint64_t) v40;
      TASSIGN(v849, v850);
      pipe_barrier(PIPE_V);
      TADD(v849, v833, v847);
      // pto: %t__tmp_v77
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v851 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v77
      ;
      uint64_t v852 = (uint64_t) v40;
      TASSIGN(v851, v852);
      pipe_barrier(PIPE_V);
      TADDS(v851, v849, v95);
      // pto: %post_ids_inline55__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v853 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %post_ids_inline55__ssa_v0
      ;
      uint64_t v854 = (uint64_t) v40;
      TASSIGN(v853, v854);
      pipe_barrier(PIPE_V);
      RoundMode v855 = RoundMode::CAST_ROUND;
      SaturationMode v856 = SaturationMode::ON;
      TCVT(v853, v851, v855, v856);
      // pto: %post_tmp_inline314__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v857 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %post_tmp_inline314__ssa_v0
      ;
      uint64_t v858 = (uint64_t) v51;
      TASSIGN(v857, v858);
      // pto: %t__tmp_v78
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v859 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v111);
      // pto: %t__tmp_v78
      ;
      uint64_t v860 = (uint64_t) v49;
      TASSIGN(v859, v860);
      // pto: %post_values_inline46__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v861 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %post_values_inline46__ssa_v0
      ;
      uint64_t v862 = (uint64_t) v52;
      TASSIGN(v861, v862);
      pipe_barrier(PIPE_V);
      TGATHER(v861, v859, v853, v857);
      // pto: %t__tmp_v79
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v863 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v70);
      // pto: %t__tmp_v79
      ;
      uint64_t v864 = (uint64_t) v40;
      TASSIGN(v863, v864);
      pipe_barrier(PIPE_V);
      TMULS(v863, v861, v93);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
      // pto: %369
      ;
      int64_t v865 = (int64_t) ((uint64_t) v719 * (uint64_t) v100);
      v863.SetValidShape(v71, v865);
      // pto: %370
      ;
      bool v866 = v691 == v84;
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
      if (v866) {
        // pto: %372
        ;
        int64_t v867 = (int64_t) ((uint64_t) v717 * (uint64_t) v100);
        // pto: %373
        ;
        int64_t v868 = v867 < v84 ? v84 : v867;
        // pto: %post_flat_inline228__ssa_v0_pview
        ;
        const int64_t v869 = 0;
        // pto: %post_flat_inline228__ssa_v0_pview
        ;
        __gm__ float* v870 = PTOAS__GLOBAL_TENSOR_DATA(v183);
        // pto: %post_flat_inline228__ssa_v0_pview
        ;
        const int64_t v871 = 1;
        // pto: %post_flat_inline228__ssa_v0_pview
        ;
        const int64_t v872 = 1;
        // pto: %post_flat_inline228__ssa_v0_pview
        ;
        const int64_t v873 = 1;
        // pto: %post_flat_inline228__ssa_v0_pview
        ;
        int64_t v874 = v71 * v175;
        // pto: %post_flat_inline228__ssa_v0_pview
        ;
        int64_t v875 = v873 * v874;
        // pto: %post_flat_inline228__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, -1> v876 = pto::Shape<1, 1, 1, 1, -1>(v871, v872, v873, v71, v865);
        // pto: %post_flat_inline228__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v877 = pto::Stride<-1, -1, -1, -1, -1>(v872 * v875, v875, v874, v175, v71);
        // pto: %post_flat_inline228__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v878 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v870 + (v869 + v84 * v175 + v868 * v71), v876, v877);
        TSTORE(v878, v863);
      };
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
      // pto: %t__tmp_v80
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v879 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %t__tmp_v80
      ;
      uint64_t v880 = (uint64_t) v44;
      TASSIGN(v879, v880);
      // pto: %374
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v881;
      // pto: %374
      ;
      uint64_t v882 = (uint64_t) v44;
      TASSIGN(v881, v882);
      // pto: %t__tmp_v81
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v883 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %t__tmp_v81
      ;
      uint64_t v884 = (uint64_t) v60;
      TASSIGN(v883, v884);
      TADDS(v883, v881, v89);
      // pto: %pre0_inline201__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v885 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v71);
      // pto: %pre0_inline201__ssa_v0
      ;
      uint64_t v886 = (uint64_t) v60;
      TASSIGN(v885, v886);
      // pto: %t__tmp_v82
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v887 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %t__tmp_v82
      ;
      uint64_t v888 = (uint64_t) v59;
      TASSIGN(v887, v888);
      // pto: %375
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v889;
      // pto: %375
      ;
      uint64_t v890 = (uint64_t) v59;
      TASSIGN(v889, v890);
      // pto: %t__tmp_v83
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v891 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %t__tmp_v83
      ;
      uint64_t v892 = (uint64_t) v61;
      TASSIGN(v891, v892);
      TADDS(v891, v889, v89);
      // pto: %pre1_inline192__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v893 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v71);
      // pto: %pre1_inline192__ssa_v0
      ;
      uint64_t v894 = (uint64_t) v61;
      TASSIGN(v893, v894);
      // pto: %t__tmp_v84
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v895 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %t__tmp_v84
      ;
      uint64_t v896 = (uint64_t) v46;
      TASSIGN(v895, v896);
      // pto: %376
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v897;
      // pto: %376
      ;
      uint64_t v898 = (uint64_t) v46;
      TASSIGN(v897, v898);
      // pto: %t__tmp_v85
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v899 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %t__tmp_v85
      ;
      uint64_t v900 = (uint64_t) v62;
      TASSIGN(v899, v900);
      TADDS(v899, v897, v89);
      // pto: %pre2_inline261__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v901 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v71);
      // pto: %pre2_inline261__ssa_v0
      ;
      uint64_t v902 = (uint64_t) v62;
      TASSIGN(v901, v902);
      // pto: %t__tmp_v86
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v903 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %t__tmp_v86
      ;
      uint64_t v904 = (uint64_t) v63;
      TASSIGN(v903, v904);
      // pto: %377
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v905;
      // pto: %377
      ;
      uint64_t v906 = (uint64_t) v63;
      TASSIGN(v905, v906);
      // pto: %t__tmp_v87
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v907 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v87);
      // pto: %t__tmp_v87
      ;
      uint64_t v908 = (uint64_t) v64;
      TASSIGN(v907, v908);
      TADDS(v907, v905, v89);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      // pto: %pre3_inline159__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v909 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v87, v71);
      // pto: %pre3_inline159__ssa_v0
      ;
      uint64_t v910 = (uint64_t) v64;
      TASSIGN(v909, v910);
      // pto: %378
      ;
      uint64_t v911 = (uint64_t) v692;
      // pto: %378
      ;
      uint64_t v912 = (uint64_t) v691;
      // pto: %381
      ;
      uint64_t v913 = (uint64_t) v691;
      // pto: %378, %379, %380, %381
      ;
      int64_t v914 = (int64_t) (v913 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) (v911 - v912) / v83) * (uint64_t) v83));
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      for (int64_t v915 = v691; v915 < v914; v915 += v83) {
        // pto: %382
        ;
        int64_t v916 = (int64_t) ((uint64_t) v915 * (uint64_t) v85);
        // pto: %385
        ;
        int64_t v917 = (int64_t) ((uint64_t) v916 + (uint64_t) v85);
        // pto: %x0_bf16_inline295__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v918 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %x0_bf16_inline295__ssa_v0
        ;
        uint64_t v919 = (uint64_t) v65;
        TASSIGN(v918, v919);
        // pto: %388
        ;
        int64_t v920 = v916 < v84 ? v84 : v916;
        // pto: %389
        ;
        const int64_t v921 = 0;
        // pto: %389
        ;
        __gm__ bfloat16_t* v922 = PTOAS__GLOBAL_TENSOR_DATA(v131);
        // pto: %389
        ;
        const int64_t v923 = 1;
        // pto: %389
        ;
        const int64_t v924 = 1;
        // pto: %389
        ;
        const int64_t v925 = 1;
        // pto: %389
        ;
        int64_t v926 = v719 * v73;
        // pto: %389
        ;
        int64_t v927 = v925 * v926;
        // pto: %389
        ;
        pto::Shape<1, 1, 1, -1, 512> v928 = pto::Shape<1, 1, 1, -1, 512>(v923, v924, v925, v719, v85);
        // pto: %389
        ;
        pto::Stride<-1, -1, -1, -1, -1> v929 = pto::Stride<-1, -1, -1, -1, -1>(v924 * v927, v927, v926, v73, v71);
        // pto: %389
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v930 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v922 + (v921 + v743 * v73 + v920 * v71), v928, v929);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v918, v930);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %x1_bf16_inline6__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v931 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %x1_bf16_inline6__ssa_v0
        ;
        uint64_t v932 = (uint64_t) v38;
        TASSIGN(v931, v932);
        // pto: %391
        ;
        int64_t v933 = (int64_t) ((uint64_t) v916 + (uint64_t) v77);
        // pto: %392
        ;
        int64_t v934 = v933 < v84 ? v84 : v933;
        // pto: %393
        ;
        const int64_t v935 = 0;
        // pto: %393
        ;
        __gm__ bfloat16_t* v936 = PTOAS__GLOBAL_TENSOR_DATA(v131);
        // pto: %393
        ;
        const int64_t v937 = 1;
        // pto: %393
        ;
        const int64_t v938 = 1;
        // pto: %393
        ;
        const int64_t v939 = 1;
        // pto: %393
        ;
        int64_t v940 = v719 * v73;
        // pto: %393
        ;
        int64_t v941 = v939 * v940;
        // pto: %393
        ;
        pto::Shape<1, 1, 1, -1, 512> v942 = pto::Shape<1, 1, 1, -1, 512>(v937, v938, v939, v719, v85);
        // pto: %393
        ;
        pto::Stride<-1, -1, -1, -1, -1> v943 = pto::Stride<-1, -1, -1, -1, -1>(v938 * v941, v941, v940, v73, v71);
        // pto: %393
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v944 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v936 + (v935 + v743 * v73 + v934 * v71), v942, v943);
        TLOAD(v931, v944);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %x2_bf16_inline5__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v945 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %x2_bf16_inline5__ssa_v0
        ;
        uint64_t v946 = (uint64_t) v66;
        TASSIGN(v945, v946);
        // pto: %395
        ;
        int64_t v947 = (int64_t) ((uint64_t) v916 + (uint64_t) v112);
        // pto: %396
        ;
        int64_t v948 = v947 < v84 ? v84 : v947;
        // pto: %397
        ;
        const int64_t v949 = 0;
        // pto: %397
        ;
        __gm__ bfloat16_t* v950 = PTOAS__GLOBAL_TENSOR_DATA(v131);
        // pto: %397
        ;
        const int64_t v951 = 1;
        // pto: %397
        ;
        const int64_t v952 = 1;
        // pto: %397
        ;
        const int64_t v953 = 1;
        // pto: %397
        ;
        int64_t v954 = v719 * v73;
        // pto: %397
        ;
        int64_t v955 = v953 * v954;
        // pto: %397
        ;
        pto::Shape<1, 1, 1, -1, 512> v956 = pto::Shape<1, 1, 1, -1, 512>(v951, v952, v953, v719, v85);
        // pto: %397
        ;
        pto::Stride<-1, -1, -1, -1, -1> v957 = pto::Stride<-1, -1, -1, -1, -1>(v952 * v955, v955, v954, v73, v71);
        // pto: %397
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v958 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v950 + (v949 + v743 * v73 + v948 * v71), v956, v957);
        TLOAD(v945, v958);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %x3_bf16_inline33__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v959 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %x3_bf16_inline33__ssa_v0
        ;
        uint64_t v960 = (uint64_t) v37;
        TASSIGN(v959, v960);
        // pto: %399
        ;
        int64_t v961 = (int64_t) ((uint64_t) v916 + (uint64_t) v113);
        // pto: %400
        ;
        int64_t v962 = v961 < v84 ? v84 : v961;
        // pto: %401
        ;
        const int64_t v963 = 0;
        // pto: %401
        ;
        __gm__ bfloat16_t* v964 = PTOAS__GLOBAL_TENSOR_DATA(v131);
        // pto: %401
        ;
        const int64_t v965 = 1;
        // pto: %401
        ;
        const int64_t v966 = 1;
        // pto: %401
        ;
        const int64_t v967 = 1;
        // pto: %401
        ;
        int64_t v968 = v719 * v73;
        // pto: %401
        ;
        int64_t v969 = v967 * v968;
        // pto: %401
        ;
        pto::Shape<1, 1, 1, -1, 512> v970 = pto::Shape<1, 1, 1, -1, 512>(v965, v966, v967, v719, v85);
        // pto: %401
        ;
        pto::Stride<-1, -1, -1, -1, -1> v971 = pto::Stride<-1, -1, -1, -1, -1>(v966 * v969, v969, v968, v73, v71);
        // pto: %401
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v972 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v964 + (v963 + v743 * v73 + v962 * v71), v970, v971);
        TLOAD(v959, v972);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %20
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v973 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %20
        ;
        uint64_t v974 = (uint64_t) v67;
        TASSIGN(v973, v974);
        // pto: %403
        ;
        int64_t v975 = v917 < v84 ? v84 : v917;
        // pto: %404
        ;
        const int64_t v976 = 0;
        // pto: %404
        ;
        __gm__ bfloat16_t* v977 = PTOAS__GLOBAL_TENSOR_DATA(v131);
        // pto: %404
        ;
        const int64_t v978 = 1;
        // pto: %404
        ;
        const int64_t v979 = 1;
        // pto: %404
        ;
        const int64_t v980 = 1;
        // pto: %404
        ;
        int64_t v981 = v719 * v73;
        // pto: %404
        ;
        int64_t v982 = v980 * v981;
        // pto: %404
        ;
        pto::Shape<1, 1, 1, -1, 512> v983 = pto::Shape<1, 1, 1, -1, 512>(v978, v979, v980, v719, v85);
        // pto: %404
        ;
        pto::Stride<-1, -1, -1, -1, -1> v984 = pto::Stride<-1, -1, -1, -1, -1>(v979 * v982, v982, v981, v73, v71);
        // pto: %404
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v985 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v977 + (v976 + v743 * v73 + v975 * v71), v983, v984);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        TLOAD(v973, v985);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        // pto: %21
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v986 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %21
        ;
        uint64_t v987 = (uint64_t) v39;
        TASSIGN(v986, v987);
        int64_t v988 = (int64_t) ((uint64_t) v916 + (uint64_t) v36);
        // pto: %407
        ;
        int64_t v989 = v988 < v84 ? v84 : v988;
        // pto: %408
        ;
        const int64_t v990 = 0;
        // pto: %408
        ;
        __gm__ bfloat16_t* v991 = PTOAS__GLOBAL_TENSOR_DATA(v131);
        // pto: %408
        ;
        const int64_t v992 = 1;
        // pto: %408
        ;
        const int64_t v993 = 1;
        // pto: %408
        ;
        const int64_t v994 = 1;
        // pto: %408
        ;
        int64_t v995 = v719 * v73;
        // pto: %408
        ;
        int64_t v996 = v994 * v995;
        // pto: %408
        ;
        pto::Shape<1, 1, 1, -1, 512> v997 = pto::Shape<1, 1, 1, -1, 512>(v992, v993, v994, v719, v85);
        // pto: %408
        ;
        pto::Stride<-1, -1, -1, -1, -1> v998 = pto::Stride<-1, -1, -1, -1, -1>(v993 * v996, v996, v995, v73, v71);
        // pto: %408
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v999 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v991 + (v990 + v743 * v73 + v989 * v71), v997, v998);
        TLOAD(v986, v999);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        // pto: %22
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1000 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %22
        ;
        uint64_t v1001 = (uint64_t) v68;
        TASSIGN(v1000, v1001);
        int64_t v1002 = (int64_t) ((uint64_t) v916 + (uint64_t) v35);
        // pto: %411
        ;
        int64_t v1003 = v1002 < v84 ? v84 : v1002;
        // pto: %412
        ;
        const int64_t v1004 = 0;
        // pto: %412
        ;
        __gm__ bfloat16_t* v1005 = PTOAS__GLOBAL_TENSOR_DATA(v131);
        // pto: %412
        ;
        const int64_t v1006 = 1;
        // pto: %412
        ;
        const int64_t v1007 = 1;
        // pto: %412
        ;
        const int64_t v1008 = 1;
        // pto: %412
        ;
        int64_t v1009 = v719 * v73;
        // pto: %412
        ;
        int64_t v1010 = v1008 * v1009;
        // pto: %412
        ;
        pto::Shape<1, 1, 1, -1, 512> v1011 = pto::Shape<1, 1, 1, -1, 512>(v1006, v1007, v1008, v719, v85);
        // pto: %412
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1012 = pto::Stride<-1, -1, -1, -1, -1>(v1007 * v1010, v1010, v1009, v73, v71);
        // pto: %412
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1013 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1005 + (v1004 + v743 * v73 + v1003 * v71), v1011, v1012);
        TLOAD(v1000, v1013);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        // pto: %23
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1014 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %23
        ;
        uint64_t v1015 = (uint64_t) v69;
        TASSIGN(v1014, v1015);
        int64_t v1016 = (int64_t) ((uint64_t) v916 + (uint64_t) v34);
        // pto: %415
        ;
        int64_t v1017 = v1016 < v84 ? v84 : v1016;
        // pto: %416
        ;
        const int64_t v1018 = 0;
        // pto: %416
        ;
        __gm__ bfloat16_t* v1019 = PTOAS__GLOBAL_TENSOR_DATA(v131);
        // pto: %416
        ;
        const int64_t v1020 = 1;
        // pto: %416
        ;
        const int64_t v1021 = 1;
        // pto: %416
        ;
        const int64_t v1022 = 1;
        // pto: %416
        ;
        int64_t v1023 = v719 * v73;
        // pto: %416
        ;
        int64_t v1024 = v1022 * v1023;
        // pto: %416
        ;
        pto::Shape<1, 1, 1, -1, 512> v1025 = pto::Shape<1, 1, 1, -1, 512>(v1020, v1021, v1022, v719, v85);
        // pto: %416
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1026 = pto::Stride<-1, -1, -1, -1, -1>(v1021 * v1024, v1024, v1023, v73, v71);
        // pto: %416
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1027 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1019 + (v1018 + v743 * v73 + v1017 * v71), v1025, v1026);
        TLOAD(v1014, v1027);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        // pto: %x0_inline7__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1028 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %x0_inline7__ssa_v0
        ;
        uint64_t v1029 = (uint64_t) v40;
        TASSIGN(v1028, v1029);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
        RoundMode v1030 = RoundMode::CAST_ROUND;
        SaturationMode v1031 = SaturationMode::ON;
        TCVT(v1028, v918, v1030, v1031);
        // pto: %x1_inline134__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1032 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %x1_inline134__ssa_v0
        ;
        uint64_t v1033 = (uint64_t) v65;
        TASSIGN(v1032, v1033);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        RoundMode v1034 = RoundMode::CAST_ROUND;
        SaturationMode v1035 = SaturationMode::ON;
        TCVT(v1032, v931, v1034, v1035);
        // pto: %x2_inline240__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1036 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %x2_inline240__ssa_v0
        ;
        uint64_t v1037 = (uint64_t) v38;
        TASSIGN(v1036, v1037);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        RoundMode v1038 = RoundMode::CAST_ROUND;
        SaturationMode v1039 = SaturationMode::ON;
        TCVT(v1036, v945, v1038, v1039);
        // pto: %x3_inline4__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1040 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %x3_inline4__ssa_v0
        ;
        uint64_t v1041 = (uint64_t) v66;
        TASSIGN(v1040, v1041);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        RoundMode v1042 = RoundMode::CAST_ROUND;
        SaturationMode v1043 = SaturationMode::ON;
        TCVT(v1040, v959, v1042, v1043);
        // pto: %y0_inline149__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1044 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %y0_inline149__ssa_v0
        ;
        uint64_t v1045 = (uint64_t) v40;
        TASSIGN(v1044, v1045);
        TROWEXPANDMUL(v1044, v1028, v885);
        // pto: %y1_inline164__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1046 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %y1_inline164__ssa_v0
        ;
        uint64_t v1047 = (uint64_t) v65;
        TASSIGN(v1046, v1047);
        TROWEXPANDMUL(v1046, v1032, v893);
        // pto: %y2_inline3__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1048 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %y2_inline3__ssa_v0
        ;
        uint64_t v1049 = (uint64_t) v38;
        TASSIGN(v1048, v1049);
        TROWEXPANDMUL(v1048, v1036, v901);
        // pto: %y3_inline2__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1050 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %y3_inline2__ssa_v0
        ;
        uint64_t v1051 = (uint64_t) v66;
        TASSIGN(v1050, v1051);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v1050, v1040, v909);
        // pto: %t__tmp_v88
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1052 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %t__tmp_v88
        ;
        uint64_t v1053 = (uint64_t) v40;
        TASSIGN(v1052, v1053);
        TADD(v1052, v1044, v1046);
        // pto: %t__tmp_v89
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1054 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %t__tmp_v89
        ;
        uint64_t v1055 = (uint64_t) v65;
        TASSIGN(v1054, v1055);
        pipe_barrier(PIPE_V);
        TADD(v1054, v1048, v1050);
        // pto: %y_tile_inline178__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1056 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %y_tile_inline178__ssa_v0
        ;
        uint64_t v1057 = (uint64_t) v40;
        TASSIGN(v1056, v1057);
        pipe_barrier(PIPE_V);
        TADD(v1056, v1052, v1054);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %y_bf16_inline1__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1058 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %y_bf16_inline1__ssa_v0
        ;
        uint64_t v1059 = (uint64_t) v40;
        TASSIGN(v1058, v1059);
        pipe_barrier(PIPE_V);
        RoundMode v1060 = RoundMode::CAST_RINT;
        SaturationMode v1061 = SaturationMode::ON;
        TCVT(v1058, v1056, v1060, v1061);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        v1058.SetValidShape(v719, v85);
        // pto: %mixed_view_inline142__ssa_v0_pview
        ;
        const int64_t v1062 = 0;
        // pto: %mixed_view_inline142__ssa_v0_pview
        ;
        __gm__ bfloat16_t* v1063 = PTOAS__GLOBAL_TENSOR_DATA(v192);
        // pto: %mixed_view_inline142__ssa_v0_pview
        ;
        const int64_t v1064 = 1;
        // pto: %mixed_view_inline142__ssa_v0_pview
        ;
        const int64_t v1065 = 1;
        // pto: %mixed_view_inline142__ssa_v0_pview
        ;
        const int64_t v1066 = 1;
        // pto: %mixed_view_inline142__ssa_v0_pview
        ;
        int64_t v1067 = v719 * v77;
        // pto: %mixed_view_inline142__ssa_v0_pview
        ;
        int64_t v1068 = v1066 * v1067;
        // pto: %mixed_view_inline142__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, -1, 512> v1069 = pto::Shape<1, 1, 1, -1, 512>(v1064, v1065, v1066, v719, v85);
        // pto: %mixed_view_inline142__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1070 = pto::Stride<-1, -1, -1, -1, -1>(v1065 * v1068, v1068, v1067, v77, v71);
        // pto: %mixed_view_inline142__ssa_v0_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1071 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1063 + (v1062 + v743 * v77 + v920 * v71), v1069, v1070);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v1071, v1058);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
        // pto: %24
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1072 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %24
        ;
        uint64_t v1073 = (uint64_t) v42;
        TASSIGN(v1072, v1073);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
        RoundMode v1074 = RoundMode::CAST_ROUND;
        SaturationMode v1075 = SaturationMode::ON;
        TCVT(v1072, v973, v1074, v1075);
        // pto: %25
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1076 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %25
        ;
        uint64_t v1077 = (uint64_t) v67;
        TASSIGN(v1076, v1077);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        RoundMode v1078 = RoundMode::CAST_ROUND;
        SaturationMode v1079 = SaturationMode::ON;
        TCVT(v1076, v986, v1078, v1079);
        // pto: %26
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1080 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %26
        ;
        uint64_t v1081 = (uint64_t) v39;
        TASSIGN(v1080, v1081);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        RoundMode v1082 = RoundMode::CAST_ROUND;
        SaturationMode v1083 = SaturationMode::ON;
        TCVT(v1080, v1000, v1082, v1083);
        // pto: %27
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1084 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %27
        ;
        uint64_t v1085 = (uint64_t) v68;
        TASSIGN(v1084, v1085);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        RoundMode v1086 = RoundMode::CAST_ROUND;
        SaturationMode v1087 = SaturationMode::ON;
        TCVT(v1084, v1014, v1086, v1087);
        // pto: %28
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1088 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %28
        ;
        uint64_t v1089 = (uint64_t) v42;
        TASSIGN(v1088, v1089);
        TROWEXPANDMUL(v1088, v1072, v885);
        // pto: %29
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1090 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %29
        ;
        uint64_t v1091 = (uint64_t) v67;
        TASSIGN(v1090, v1091);
        TROWEXPANDMUL(v1090, v1076, v893);
        // pto: %30
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1092 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %30
        ;
        uint64_t v1093 = (uint64_t) v39;
        TASSIGN(v1092, v1093);
        TROWEXPANDMUL(v1092, v1080, v901);
        // pto: %31
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1094 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %31
        ;
        uint64_t v1095 = (uint64_t) v68;
        TASSIGN(v1094, v1095);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v1094, v1084, v909);
        // pto: %32
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1096 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %32
        ;
        uint64_t v1097 = (uint64_t) v42;
        TASSIGN(v1096, v1097);
        TADD(v1096, v1088, v1090);
        // pto: %33
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1098 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %33
        ;
        uint64_t v1099 = (uint64_t) v67;
        TASSIGN(v1098, v1099);
        pipe_barrier(PIPE_V);
        TADD(v1098, v1092, v1094);
        // pto: %34
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1100 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %34
        ;
        uint64_t v1101 = (uint64_t) v42;
        TASSIGN(v1100, v1101);
        pipe_barrier(PIPE_V);
        TADD(v1100, v1096, v1098);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        // pto: %35
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1102 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %35
        ;
        uint64_t v1103 = (uint64_t) v42;
        TASSIGN(v1102, v1103);
        pipe_barrier(PIPE_V);
        RoundMode v1104 = RoundMode::CAST_RINT;
        SaturationMode v1105 = SaturationMode::ON;
        TCVT(v1102, v1100, v1104, v1105);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        v1102.SetValidShape(v719, v85);
        // pto: %422
        ;
        const int64_t v1106 = 0;
        // pto: %422
        ;
        __gm__ bfloat16_t* v1107 = PTOAS__GLOBAL_TENSOR_DATA(v192);
        // pto: %422
        ;
        const int64_t v1108 = 1;
        // pto: %422
        ;
        const int64_t v1109 = 1;
        // pto: %422
        ;
        const int64_t v1110 = 1;
        // pto: %422
        ;
        int64_t v1111 = v719 * v77;
        // pto: %422
        ;
        int64_t v1112 = v1110 * v1111;
        // pto: %422
        ;
        pto::Shape<1, 1, 1, -1, 512> v1113 = pto::Shape<1, 1, 1, -1, 512>(v1108, v1109, v1110, v719, v85);
        // pto: %422
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1114 = pto::Stride<-1, -1, -1, -1, -1>(v1109 * v1112, v1112, v1111, v77, v71);
        // pto: %422
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1115 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1107 + (v1106 + v743 * v77 + v975 * v71), v1113, v1114);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v1115, v1102);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
      };
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
      // pto: %423
      ;
      uint64_t v1116 = (uint64_t) v692;
      // pto: %423, %424
      ;
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
      if ((int64_t) (v1116 - (uint64_t) v914) == v71) {
        // pto: %425
        ;
        int64_t v1117 = (int64_t) ((uint64_t) v914 * (uint64_t) v85);
        // pto: %37
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1118 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %37
        ;
        uint64_t v1119 = (uint64_t) v65;
        TASSIGN(v1118, v1119);
        // pto: %428
        ;
        int64_t v1120 = v1117 < v84 ? v84 : v1117;
        // pto: %429
        ;
        const int64_t v1121 = 0;
        // pto: %429
        ;
        __gm__ bfloat16_t* v1122 = PTOAS__GLOBAL_TENSOR_DATA(v131);
        // pto: %429
        ;
        const int64_t v1123 = 1;
        // pto: %429
        ;
        const int64_t v1124 = 1;
        // pto: %429
        ;
        const int64_t v1125 = 1;
        // pto: %429
        ;
        int64_t v1126 = v719 * v73;
        // pto: %429
        ;
        int64_t v1127 = v1125 * v1126;
        // pto: %429
        ;
        pto::Shape<1, 1, 1, -1, 512> v1128 = pto::Shape<1, 1, 1, -1, 512>(v1123, v1124, v1125, v719, v85);
        // pto: %429
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1129 = pto::Stride<-1, -1, -1, -1, -1>(v1124 * v1127, v1127, v1126, v73, v71);
        // pto: %429
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1130 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1122 + (v1121 + v743 * v73 + v1120 * v71), v1128, v1129);
        TLOAD(v1118, v1130);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %38
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1131 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %38
        ;
        uint64_t v1132 = (uint64_t) v40;
        TASSIGN(v1131, v1132);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v1133 = RoundMode::CAST_ROUND;
        SaturationMode v1134 = SaturationMode::ON;
        TCVT(v1131, v1118, v1133, v1134);
        // pto: %39
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1135 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %39
        ;
        uint64_t v1136 = (uint64_t) v38;
        TASSIGN(v1135, v1136);
        // pto: %431
        ;
        int64_t v1137 = (int64_t) ((uint64_t) v1117 + (uint64_t) v77);
        // pto: %432
        ;
        int64_t v1138 = v1137 < v84 ? v84 : v1137;
        // pto: %433
        ;
        const int64_t v1139 = 0;
        // pto: %433
        ;
        __gm__ bfloat16_t* v1140 = PTOAS__GLOBAL_TENSOR_DATA(v131);
        // pto: %433
        ;
        const int64_t v1141 = 1;
        // pto: %433
        ;
        const int64_t v1142 = 1;
        // pto: %433
        ;
        const int64_t v1143 = 1;
        // pto: %433
        ;
        int64_t v1144 = v719 * v73;
        // pto: %433
        ;
        int64_t v1145 = v1143 * v1144;
        // pto: %433
        ;
        pto::Shape<1, 1, 1, -1, 512> v1146 = pto::Shape<1, 1, 1, -1, 512>(v1141, v1142, v1143, v719, v85);
        // pto: %433
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1147 = pto::Stride<-1, -1, -1, -1, -1>(v1142 * v1145, v1145, v1144, v73, v71);
        // pto: %433
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1148 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1140 + (v1139 + v743 * v73 + v1138 * v71), v1146, v1147);
        TLOAD(v1135, v1148);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %40
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1149 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %40
        ;
        uint64_t v1150 = (uint64_t) v65;
        TASSIGN(v1149, v1150);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v1151 = RoundMode::CAST_ROUND;
        SaturationMode v1152 = SaturationMode::ON;
        TCVT(v1149, v1135, v1151, v1152);
        // pto: %41
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1153 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %41
        ;
        uint64_t v1154 = (uint64_t) v66;
        TASSIGN(v1153, v1154);
        // pto: %435
        ;
        int64_t v1155 = (int64_t) ((uint64_t) v1117 + (uint64_t) v112);
        // pto: %436
        ;
        int64_t v1156 = v1155 < v84 ? v84 : v1155;
        // pto: %437
        ;
        const int64_t v1157 = 0;
        // pto: %437
        ;
        __gm__ bfloat16_t* v1158 = PTOAS__GLOBAL_TENSOR_DATA(v131);
        // pto: %437
        ;
        const int64_t v1159 = 1;
        // pto: %437
        ;
        const int64_t v1160 = 1;
        // pto: %437
        ;
        const int64_t v1161 = 1;
        // pto: %437
        ;
        int64_t v1162 = v719 * v73;
        // pto: %437
        ;
        int64_t v1163 = v1161 * v1162;
        // pto: %437
        ;
        pto::Shape<1, 1, 1, -1, 512> v1164 = pto::Shape<1, 1, 1, -1, 512>(v1159, v1160, v1161, v719, v85);
        // pto: %437
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1165 = pto::Stride<-1, -1, -1, -1, -1>(v1160 * v1163, v1163, v1162, v73, v71);
        // pto: %437
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1166 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1158 + (v1157 + v743 * v73 + v1156 * v71), v1164, v1165);
        TLOAD(v1153, v1166);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %42
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1167 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %42
        ;
        uint64_t v1168 = (uint64_t) v38;
        TASSIGN(v1167, v1168);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v1169 = RoundMode::CAST_ROUND;
        SaturationMode v1170 = SaturationMode::ON;
        TCVT(v1167, v1153, v1169, v1170);
        // pto: %43
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1171 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %43
        ;
        uint64_t v1172 = (uint64_t) v42;
        TASSIGN(v1171, v1172);
        // pto: %439
        ;
        int64_t v1173 = (int64_t) ((uint64_t) v1117 + (uint64_t) v113);
        // pto: %440
        ;
        int64_t v1174 = v1173 < v84 ? v84 : v1173;
        // pto: %441
        ;
        const int64_t v1175 = 0;
        // pto: %441
        ;
        __gm__ bfloat16_t* v1176 = PTOAS__GLOBAL_TENSOR_DATA(v131);
        // pto: %441
        ;
        const int64_t v1177 = 1;
        // pto: %441
        ;
        const int64_t v1178 = 1;
        // pto: %441
        ;
        const int64_t v1179 = 1;
        // pto: %441
        ;
        int64_t v1180 = v719 * v73;
        // pto: %441
        ;
        int64_t v1181 = v1179 * v1180;
        // pto: %441
        ;
        pto::Shape<1, 1, 1, -1, 512> v1182 = pto::Shape<1, 1, 1, -1, 512>(v1177, v1178, v1179, v719, v85);
        // pto: %441
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1183 = pto::Stride<-1, -1, -1, -1, -1>(v1178 * v1181, v1181, v1180, v73, v71);
        // pto: %441
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1184 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1176 + (v1175 + v743 * v73 + v1174 * v71), v1182, v1183);
        TLOAD(v1171, v1184);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %44
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1185 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %44
        ;
        uint64_t v1186 = (uint64_t) v66;
        TASSIGN(v1185, v1186);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v1187 = RoundMode::CAST_ROUND;
        SaturationMode v1188 = SaturationMode::ON;
        TCVT(v1185, v1171, v1187, v1188);
        // pto: %45
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1189 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %45
        ;
        uint64_t v1190 = (uint64_t) v40;
        TASSIGN(v1189, v1190);
        TROWEXPANDMUL(v1189, v1131, v885);
        // pto: %46
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1191 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %46
        ;
        uint64_t v1192 = (uint64_t) v65;
        TASSIGN(v1191, v1192);
        TROWEXPANDMUL(v1191, v1149, v893);
        // pto: %47
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1193 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %47
        ;
        uint64_t v1194 = (uint64_t) v38;
        TASSIGN(v1193, v1194);
        TROWEXPANDMUL(v1193, v1167, v901);
        // pto: %48
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1195 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %48
        ;
        uint64_t v1196 = (uint64_t) v66;
        TASSIGN(v1195, v1196);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v1195, v1185, v909);
        // pto: %49
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1197 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %49
        ;
        uint64_t v1198 = (uint64_t) v40;
        TASSIGN(v1197, v1198);
        TADD(v1197, v1189, v1191);
        // pto: %50
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1199 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %50
        ;
        uint64_t v1200 = (uint64_t) v65;
        TASSIGN(v1199, v1200);
        pipe_barrier(PIPE_V);
        TADD(v1199, v1193, v1195);
        // pto: %51
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1201 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %51
        ;
        uint64_t v1202 = (uint64_t) v40;
        TASSIGN(v1201, v1202);
        pipe_barrier(PIPE_V);
        TADD(v1201, v1197, v1199);
        // pto: %52
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1203 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v719, v85);
        // pto: %52
        ;
        uint64_t v1204 = (uint64_t) v40;
        TASSIGN(v1203, v1204);
        pipe_barrier(PIPE_V);
        RoundMode v1205 = RoundMode::CAST_RINT;
        SaturationMode v1206 = SaturationMode::ON;
        TCVT(v1203, v1201, v1205, v1206);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        v1203.SetValidShape(v719, v85);
        // pto: %445
        ;
        const int64_t v1207 = 0;
        // pto: %445
        ;
        __gm__ bfloat16_t* v1208 = PTOAS__GLOBAL_TENSOR_DATA(v192);
        // pto: %445
        ;
        const int64_t v1209 = 1;
        // pto: %445
        ;
        const int64_t v1210 = 1;
        // pto: %445
        ;
        const int64_t v1211 = 1;
        // pto: %445
        ;
        int64_t v1212 = v719 * v77;
        // pto: %445
        ;
        int64_t v1213 = v1211 * v1212;
        // pto: %445
        ;
        pto::Shape<1, 1, 1, -1, 512> v1214 = pto::Shape<1, 1, 1, -1, 512>(v1209, v1210, v1211, v719, v85);
        // pto: %445
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1215 = pto::Stride<-1, -1, -1, -1, -1>(v1210 * v1213, v1213, v1212, v77, v71);
        // pto: %445
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1216 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1208 + (v1207 + v743 * v77 + v1120 * v71), v1214, v1215);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v1216, v1203);
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