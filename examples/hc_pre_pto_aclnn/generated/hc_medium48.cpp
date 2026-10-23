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

AICORE void hc_medium48_aic(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ bfloat16_t* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, __gm__ float* v10, __gm__ float* v11, __gm__ bfloat16_t* v12, __gm__ float* v13, int64_t v14, int32_t v15, int64_t v16, int64_t v17, int64_t v18, int64_t v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int64_t v28, int64_t v29, int64_t v30, int32_t v31, int32_t v32) {
  using T = float;

  #if defined(__DAV_CUBE__)
  const int64_t v33 = 384;
  const int32_t v34 = 0;
  // pto: %c0_i64
  const int64_t v35 = 0;
  // pto: %c196608_i64
  const int64_t v36 = 196608;
  // pto: %c24576_i64
  const int64_t v37 = 24576;
  // pto: %c16384_i64
  const int64_t v38 = 16384;
  // pto: %c32_index
  const int64_t v39 = 32;
  // pto: %c1_index
  const int64_t v40 = 1;
  // pto: %c24_index
  const int64_t v41 = 24;
  // pto: %c16384_index
  const int64_t v42 = 16384;
  // pto: %c16_index
  const int64_t v43 = 16;
  // pto: %c48_index
  const int64_t v44 = 48;
  // pto: %c1024_index
  const int64_t v45 = 1024;
  // pto: %c0_index
  const int64_t v46 = 0;
  // pto: %c2_index
  const int64_t v47 = 2;
  // pto: %c512_index
  const int64_t v48 = 512;
  // pto: %c256_index
  const int64_t v49 = 256;
  // pto: %c128_index
  const int64_t v50 = 128;
  // pto: %cn128_index
  const int64_t v51 = -128;
  // pto: %41
  int64_t v52 = (int64_t) ((uint64_t) v29 * (uint64_t) v43);
  // pto: %mixes_partials_inline376__ssa_v0_view
  const int64_t v53 = 1;
  // pto: %mixes_partials_inline376__ssa_v0_view
  const int64_t v54 = 1;
  // pto: %mixes_partials_inline376__ssa_v0_view
  const int64_t v55 = 1;
  // pto: %mixes_partials_inline376__ssa_v0_view
  int64_t v56 = v52 * v39;
  // pto: %mixes_partials_inline376__ssa_v0_view
  int64_t v57 = v55 * v56;
  // pto: %mixes_partials_inline376__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v58 = pto::Shape<1, 1, 1, -1, -1>(v53, v54, v55, v52, v39);
  // pto: %mixes_partials_inline376__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v59 = pto::Stride<-1, -1, -1, -1, -1>(v54 * v57, v57, v56, v39, v40);
  // pto: %mixes_partials_inline376__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v60 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v58, v59);
  // pto: %hc_fn__ssa_v0_view
  const int64_t v61 = 1;
  // pto: %hc_fn__ssa_v0_view
  const int64_t v62 = 1;
  // pto: %hc_fn__ssa_v0_view
  const int64_t v63 = 1;
  // pto: %hc_fn__ssa_v0_view
  int64_t v64 = v41 * v42;
  // pto: %hc_fn__ssa_v0_view
  int64_t v65 = v63 * v64;
  // pto: %hc_fn__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v66 = pto::Shape<1, 1, 1, -1, -1>(v61, v62, v63, v41, v42);
  // pto: %hc_fn__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v67 = pto::Stride<-1, -1, -1, -1, -1>(v62 * v65, v65, v64, v42, v40);
  // pto: %hc_fn__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v68 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v66, v67);
  // pto: %mid_stream_product_inline395__phi_v10
  Tile<TileType::Acc, float, 48, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v69 = Tile<TileType::Acc, float, 48, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v44, v41);
  // pto: %mid_stream_product_inline395__phi_v10
  uint64_t v70 = (uint64_t) v35;
  TASSIGN(v69, v70);
  const int32_t v71 = 0;
  auto v72 = TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>(v13, v71, v34);
  // pto: %worker_inline354__ssa_v0
  // pto: %linear_workers_inline519__ssa_v0_idx
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  for (int64_t v73 = (int64_t) v31; v73 < v14; v73 += (int64_t) v15) {
    // pto: %13
    ;
    int64_t v74 = v73 % v43;
    // pto: %15, %16, %11, %12
    ;
    int64_t v75 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v74 * (uint64_t) v16) + (uint64_t) ((int64_t) (uint64_t) (v73 / v43) * (uint64_t) v44));
    // pto: %mid_stream_product_inline395__tile
    ;
    Tile<TileType::Acc, float, 48, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v76 = Tile<TileType::Acc, float, 48, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v44, v39);
    // pto: %mid_stream_product_inline395__tile
    ;
    uint64_t v77 = (uint64_t) v35;
    TASSIGN(v76, v77);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int64_t v78 = v46; v78 < v47; v78 += v40) {
      // pto: %14, %18, %17
      ;
      int64_t v79 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v74 * (uint64_t) v45) + (uint64_t) ((int64_t) (uint64_t) v78 * (uint64_t) v48));
      // pto: %mid_weight_chunk_inline389__tile
      ;
      Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v80 = Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v41, v48);
      // pto: %mid_weight_chunk_inline389__tile
      ;
      uint64_t v81 = (uint64_t) v36;
      TASSIGN(v80, v81);
      // pto: %19
      ;
      int64_t v82 = v79 < v46 ? v46 : v79;
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v83 = 0;
      // pto: %hc_fn__ssa_v0_pview
      ;
      __gm__ float* v84 = PTOAS__GLOBAL_TENSOR_DATA(v68);
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v85 = 1;
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v86 = 1;
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v87 = 1;
      // pto: %hc_fn__ssa_v0_pview
      ;
      int64_t v88 = v41 * v42;
      // pto: %hc_fn__ssa_v0_pview
      ;
      int64_t v89 = v87 * v88;
      // pto: %hc_fn__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 24, -1> v90 = pto::Shape<1, 1, 1, 24, -1>(v85, v86, v87, v41, v48);
      // pto: %hc_fn__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v91 = pto::Stride<-1, -1, -1, -1, -1>(v86 * v89, v89, v88, v42, v40);
      // pto: %hc_fn__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v92 = GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v84 + (v83 + v46 * v42 + v82 * v40), v90, v91);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v80, v92);
      // pto: %mid_gathered_chunk_inline563__tile
      ;
      const int32_t v93 = 48;
      // pto: %mid_gathered_chunk_inline563__tile
      ;
      const int32_t v94 = 512;
      // pto: %mid_gathered_chunk_inline563__tile
      ;
      Tile<TileType::Mat, float, 48, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v95 = Tile<TileType::Mat, float, 48, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v93, v94);
      pipe_barrier(PIPE_MTE2);
      TPOP<TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>, Tile<TileType::Mat, float, 48, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v72, v95);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %20
      ;
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      if (v78 == v46) {
        // pto: %mid_weight_chunk_inline389__tile_t
        ;
        Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v48, v41);
        // pto: %mid_weight_chunk_inline389__tile_t
        ;
        uint64_t v97 = (uint64_t) v36;
        TASSIGN(v96, v97);
        // pto: %mid_stream_product_inline395__tile_l0_init_storage
        ;
        Tile<TileType::Acc, float, 48, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v98 = Tile<TileType::Acc, float, 48, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v44, v39);
        // pto: %mid_stream_product_inline395__tile_l0_init_storage
        ;
        uint64_t v99 = (uint64_t) v35;
        TASSIGN(v98, v99);
        v98.SetValidShape(v44, v41);
        for (int64_t v100 = v46; v100 < v48; v100 += v49) {
          // pto: %mid_stream_product_inline395__tile_l0_a
          ;
          Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v101 = Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v44, v50);
          // pto: %mid_stream_product_inline395__tile_l0_a
          ;
          uint64_t v102 = (uint64_t) v35;
          TASSIGN(v101, v102);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          pipe_barrier(PIPE_MTE1);
          uint16_t v103 = static_cast<uint16_t>(v46);
          uint16_t v104 = static_cast<uint16_t>(v100);
          TEXTRACT(v101, v95, v103, v104);
          // pto: %21
          ;
          int64_t v105 = (int64_t) ((uint64_t) v48 - (uint64_t) v100);
          // pto: %22
          ;
          int64_t v106 = v105 < v50 ? v105 : v50;
          // pto: %mid_stream_product_inline395__tile_l0_b
          ;
          Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v107 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v106, v41);
          // pto: %mid_stream_product_inline395__tile_l0_b
          ;
          uint64_t v108 = (uint64_t) v35;
          TASSIGN(v107, v108);
          uint16_t v109 = static_cast<uint16_t>(v100);
          uint16_t v110 = static_cast<uint16_t>(v46);
          TEXTRACT(v107, v96, v109, v110);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          // pto: %0
          ;
          Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v111 = Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v44, v50);
          // pto: %0
          ;
          uint64_t v112 = (uint64_t) v37;
          TASSIGN(v111, v112);
          // pto: %23
          ;
          int64_t v113 = (int64_t) ((uint64_t) v100 + (uint64_t) v50);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
          uint16_t v114 = static_cast<uint16_t>(v46);
          uint16_t v115 = static_cast<uint16_t>(v113);
          TEXTRACT(v111, v95, v114, v115);
          int64_t v116 = (int64_t) ((uint64_t) v33 - (uint64_t) v100);
          // pto: %26
          ;
          int64_t v117 = v116 < v46 ? v46 : v116;
          // pto: %27
          ;
          int64_t v118 = v117 < v50 ? v117 : v50;
          // pto: %1
          ;
          Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v119 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v118, v41);
          // pto: %1
          ;
          uint64_t v120 = (uint64_t) v38;
          TASSIGN(v119, v120);
          uint16_t v121 = static_cast<uint16_t>(v113);
          uint16_t v122 = static_cast<uint16_t>(v46);
          TEXTRACT(v119, v96, v121, v122);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          // pto: %29
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          if (v100 == v46) {
            pipe_barrier(PIPE_M);
            TMATMUL(v98, v101, v107);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v98, v98, v101, v107);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          // pto: %30
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          if (v100 == v51) {
            pipe_barrier(PIPE_M);
            TMATMUL(v98, v111, v119);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v98, v98, v111, v119);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        };
      } else {
        // pto: %3
        ;
        Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v123 = Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v48, v41);
        // pto: %3
        ;
        uint64_t v124 = (uint64_t) v36;
        TASSIGN(v123, v124);
        for (int64_t v125 = v46; v125 < v48; v125 += v49) {
          // pto: %4
          ;
          Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v126 = Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v44, v50);
          // pto: %4
          ;
          uint64_t v127 = (uint64_t) v35;
          TASSIGN(v126, v127);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
          uint16_t v128 = static_cast<uint16_t>(v46);
          uint16_t v129 = static_cast<uint16_t>(v125);
          TEXTRACT(v126, v95, v128, v129);
          // pto: %32
          ;
          int64_t v130 = (int64_t) ((uint64_t) v48 - (uint64_t) v125);
          // pto: %33
          ;
          int64_t v131 = v130 < v50 ? v130 : v50;
          // pto: %5
          ;
          Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v132 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v131, v41);
          // pto: %5
          ;
          uint64_t v133 = (uint64_t) v35;
          TASSIGN(v132, v133);
          uint16_t v134 = static_cast<uint16_t>(v125);
          uint16_t v135 = static_cast<uint16_t>(v46);
          TEXTRACT(v132, v123, v134, v135);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
          // pto: %6
          ;
          Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v136 = Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v44, v50);
          // pto: %6
          ;
          uint64_t v137 = (uint64_t) v37;
          TASSIGN(v136, v137);
          // pto: %34
          ;
          int64_t v138 = (int64_t) ((uint64_t) v125 + (uint64_t) v50);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
          uint16_t v139 = static_cast<uint16_t>(v46);
          uint16_t v140 = static_cast<uint16_t>(v138);
          TEXTRACT(v136, v95, v139, v140);
          int64_t v141 = (int64_t) ((uint64_t) v33 - (uint64_t) v125);
          // pto: %37
          ;
          int64_t v142 = v141 < v46 ? v46 : v141;
          // pto: %38
          ;
          int64_t v143 = v142 < v50 ? v142 : v50;
          // pto: %7
          ;
          Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v144 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v143, v41);
          // pto: %7
          ;
          uint64_t v145 = (uint64_t) v38;
          TASSIGN(v144, v145);
          uint16_t v146 = static_cast<uint16_t>(v138);
          uint16_t v147 = static_cast<uint16_t>(v46);
          TEXTRACT(v144, v123, v146, v147);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
          pipe_barrier(PIPE_M);
          TMATMUL_ACC(v76, v76, v126, v132);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
          pipe_barrier(PIPE_M);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
          TMATMUL_ACC(v76, v76, v136, v144);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        };
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TFREE<TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>, TileSplitAxis::TILE_UP_DOWN>(v72);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    v76.SetValidShape(v44, v39);
    // pto: %40
    ;
    int64_t v148 = v75 < v46 ? v46 : v75;
    // pto: %mixes_partials_inline376__iter_v1_pview
    ;
    __gm__ float* v149 = PTOAS__GLOBAL_TENSOR_DATA(v60);
    // pto: %mixes_partials_inline376__iter_v1_pview
    ;
    const int64_t v150 = 0;
    // pto: %mixes_partials_inline376__iter_v1_pview
    ;
    const int64_t v151 = 32;
    // pto: %mixes_partials_inline376__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 48, 32> v152 = pto::Shape<1, 1, 1, 48, 32>();
    // pto: %mixes_partials_inline376__iter_v1_pview
    ;
    pto::Stride<1536, 1536, 1536, 32, 1> v153 = pto::Stride<1536, 1536, 1536, 32, 1>();
    // pto: %mixes_partials_inline376__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 48, 32>, pto::Stride<1536, 1536, 1536, 32, 1>, pto::Layout::ND> v154 = GlobalTensor<float, pto::Shape<1, 1, 1, 48, 32>, pto::Stride<1536, 1536, 1536, 32, 1>, pto::Layout::ND>(v149 + (v150 + v148 * v151), v152, v153);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v154, v76);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
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

AICORE void hc_medium48_aiv(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ bfloat16_t* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, __gm__ float* v10, __gm__ float* v11, __gm__ bfloat16_t* v12, __gm__ float* v13, int64_t v14, int32_t v15, int64_t v16, int64_t v17, int64_t v18, int64_t v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int64_t v28, int64_t v29, int64_t v30, int32_t v31, int32_t v32, int32_t v33) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c256_i64
  const int64_t v34 = 256;
  // pto: %c65792_i64
  const int64_t v35 = 65792;
  // pto: %c16640_i64
  const int64_t v36 = 16640;
  // pto: %c352_i64
  const int64_t v37 = 352;
  // pto: %c33024_i64
  const int64_t v38 = 33024;
  // pto: %c17664_i64
  const int64_t v39 = 17664;
  // pto: %c18176_i64
  const int64_t v40 = 18176;
  // pto: %c18688_i64
  const int64_t v41 = 18688;
  // pto: %c17728_i64
  const int64_t v42 = 17728;
  const int64_t v43 = 16672;
  // pto: %c17152_i64
  const int64_t v44 = 17152;
  // pto: %c17408_i64
  const int64_t v45 = 17408;
  // pto: %c17920_i64
  const int64_t v46 = 17920;
  // pto: %c16768_i64
  const int64_t v47 = 16768;
  // pto: %c16896_i64
  const int64_t v48 = 16896;
  // pto: %c17024_i64
  const int64_t v49 = 17024;
  // pto: %c17280_i64
  const int64_t v50 = 17280;
  // pto: %c18048_i64
  const int64_t v51 = 18048;
  // pto: %c17536_i64
  const int64_t v52 = 17536;
  // pto: %c18432_i64
  const int64_t v53 = 18432;
  // pto: %c18464_i64
  const int64_t v54 = 18464;
  // pto: %c18496_i64
  const int64_t v55 = 18496;
  // pto: %c18528_i64
  const int64_t v56 = 18528;
  // pto: %c17696_i64
  const int64_t v57 = 17696;
  // pto: %c0_i64
  const int64_t v58 = 0;
  // pto: %c49408_i64
  const int64_t v59 = 49408;
  // pto: %c57600_i64
  const int64_t v60 = 57600;
  // pto: %c82176_i64
  const int64_t v61 = 82176;
  // pto: %c98560_i64
  const int64_t v62 = 98560;
  // pto: %c106752_i64
  const int64_t v63 = 106752;
  // pto: %c8448_i64
  const int64_t v64 = 8448;
  // pto: %c32_index
  const int64_t v65 = 32;
  // pto: %c1_index
  const int64_t v66 = 1;
  // pto: %c24_index
  const int64_t v67 = 24;
  // pto: %c16384_index
  const int64_t v68 = 16384;
  // pto: %c3_index
  const int64_t v69 = 3;
  const int64_t v70 = 16;
  // pto: %c4096_index
  const int64_t v71 = 4096;
  // pto: %c0_i32
  const int32_t v72 = 0;
  // pto: %c48_index
  const int64_t v73 = 48;
  // pto: %c1024_index
  const int64_t v74 = 1024;
  // pto: %cst_41
  const float v75 = 0.0f;
  // pto: %c0_index
  const int64_t v76 = 0;
  // pto: %c2_index
  const int64_t v77 = 2;
  // pto: %c512_index
  const int64_t v78 = 512;
  // pto: %c256_index
  const int64_t v79 = 256;
  // pto: %c8_index
  const int64_t v80 = 8;
  // pto: %cst_47
  const float v81 = 6.10351563E-5f;
  // pto: %cst_48
  const float v82 = 9.99999997E-7f;
  // pto: %c192_index
  const int64_t v83 = 192;
  // pto: %c64_index
  const int64_t v84 = 64;
  // pto: %c1_i32
  const int32_t v85 = 1;
  // pto: %c2_i32
  const int32_t v86 = 2;
  // pto: %c4_i32
  const int32_t v87 = 4;
  // pto: %c8_i32
  const int32_t v88 = 8;
  // pto: %c19_index
  const int64_t v89 = 19;
  // pto: %cst_56
  const float v90 = 0.25f;
  // pto: %cst_57
  const float v91 = 1.0f;
  // pto: %cst_58
  const float v92 = 4.0f;
  // pto: %cst_59
  const float v93 = 2.0f;
  // pto: %c4_index
  const int64_t v94 = 4;
  // pto: %c8192_index
  const int64_t v95 = 8192;
  // pto: %c12288_index
  const int64_t v96 = 12288;
  // pto: %255
  int64_t v97 = (int64_t) ((uint64_t) v29 * (uint64_t) v70);
  // pto: %rms_partials_inline548__ssa_v0_view
  const int64_t v98 = 1;
  // pto: %rms_partials_inline548__ssa_v0_view
  const int64_t v99 = 1;
  // pto: %rms_partials_inline548__ssa_v0_view
  const int64_t v100 = 1;
  // pto: %rms_partials_inline548__ssa_v0_view
  int64_t v101 = v97 * v66;
  // pto: %rms_partials_inline548__ssa_v0_view
  int64_t v102 = v100 * v101;
  // pto: %rms_partials_inline548__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v103 = pto::Shape<1, 1, 1, -1, -1>(v98, v99, v100, v97, v66);
  // pto: %rms_partials_inline548__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v104 = pto::Stride<-1, -1, -1, -1, -1>(v99 * v102, v102, v101, v66, v97);
  // pto: %rms_partials_inline548__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v105 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v103, v104);
  // pto: %x_flat_inline459__ssa_v0_view
  const int64_t v106 = 1;
  // pto: %x_flat_inline459__ssa_v0_view
  const int64_t v107 = 1;
  // pto: %x_flat_inline459__ssa_v0_view
  const int64_t v108 = 1;
  // pto: %x_flat_inline459__ssa_v0_view
  int64_t v109 = (int64_t) v30;
  // pto: %x_flat_inline459__ssa_v0_view
  int64_t v110 = v109 * v68;
  // pto: %x_flat_inline459__ssa_v0_view
  int64_t v111 = v108 * v110;
  // pto: %x_flat_inline459__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v112 = pto::Shape<1, 1, 1, -1, -1>(v106, v107, v108, v109, v68);
  // pto: %x_flat_inline459__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v113 = pto::Stride<-1, -1, -1, -1, -1>(v107 * v111, v111, v110, v68, v66);
  // pto: %x_flat_inline459__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v114 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v112, v113);
  // pto: %256
  int64_t v115 = (int64_t) ((uint64_t) v29 * (uint64_t) v65);
  // pto: %parts_2d_inline445__ssa_v0_view
  const int64_t v116 = 1;
  // pto: %parts_2d_inline445__ssa_v0_view
  const int64_t v117 = 1;
  // pto: %parts_2d_inline445__ssa_v0_view
  const int64_t v118 = 1;
  // pto: %parts_2d_inline445__ssa_v0_view
  int64_t v119 = v70 * v115;
  // pto: %parts_2d_inline445__ssa_v0_view
  int64_t v120 = v118 * v119;
  // pto: %parts_2d_inline445__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v121 = pto::Shape<1, 1, 1, -1, -1>(v116, v117, v118, v70, v115);
  // pto: %parts_2d_inline445__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v122 = pto::Stride<-1, -1, -1, -1, -1>(v117 * v120, v120, v119, v115, v66);
  // pto: %parts_2d_inline445__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v123 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v121, v122);
  // pto: %rms_2d_inline408__ssa_v0_view
  const int64_t v124 = 1;
  // pto: %rms_2d_inline408__ssa_v0_view
  const int64_t v125 = 1;
  // pto: %rms_2d_inline408__ssa_v0_view
  const int64_t v126 = 1;
  // pto: %rms_2d_inline408__ssa_v0_view
  int64_t v127 = (int64_t) v29;
  // pto: %rms_2d_inline408__ssa_v0_view
  int64_t v128 = v70 * v127;
  // pto: %rms_2d_inline408__ssa_v0_view
  int64_t v129 = v126 * v128;
  // pto: %rms_2d_inline408__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v130 = pto::Shape<1, 1, 1, -1, -1>(v124, v125, v126, v70, (int64_t) v29);
  // pto: %rms_2d_inline408__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v131 = pto::Stride<-1, -1, -1, -1, -1>(v125 * v129, v129, v128, v127, v66);
  // pto: %rms_2d_inline408__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v132 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v130, v131);
  // pto: %hc_base_2d_inline576__ssa_v0_view
  const int64_t v133 = 1;
  // pto: %hc_base_2d_inline576__ssa_v0_view
  const int64_t v134 = 1;
  // pto: %hc_base_2d_inline576__ssa_v0_view
  const int64_t v135 = 1;
  // pto: %hc_base_2d_inline576__ssa_v0_view
  int64_t v136 = v66 * v67;
  // pto: %hc_base_2d_inline576__ssa_v0_view
  int64_t v137 = v135 * v136;
  // pto: %hc_base_2d_inline576__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v138 = pto::Shape<1, 1, 1, -1, -1>(v133, v134, v135, v66, v67);
  // pto: %hc_base_2d_inline576__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v139 = pto::Stride<-1, -1, -1, -1, -1>(v134 * v137, v137, v136, v67, v66);
  // pto: %hc_base_2d_inline576__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v140 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v8, v138, v139);
  // pto: %comb_inline367__ssa_v0_view
  const int64_t v141 = 1;
  // pto: %comb_inline367__ssa_v0_view
  const int64_t v142 = 1;
  // pto: %comb_inline367__ssa_v0_view
  const int64_t v143 = 1;
  // pto: %comb_inline367__ssa_v0_view
  int64_t v144 = (int64_t) v30;
  // pto: %comb_inline367__ssa_v0_view
  int64_t v145 = v144 * v70;
  // pto: %comb_inline367__ssa_v0_view
  int64_t v146 = v143 * v145;
  // pto: %comb_inline367__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v147 = pto::Shape<1, 1, 1, -1, -1>(v141, v142, v143, v144, v70);
  // pto: %comb_inline367__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v148 = pto::Stride<-1, -1, -1, -1, -1>(v142 * v146, v146, v145, v70, v66);
  // pto: %comb_inline367__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v149 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v9, v147, v148);
  // pto: %hc_scale_2d_inline476__ssa_v0_view
  const int64_t v150 = 1;
  // pto: %hc_scale_2d_inline476__ssa_v0_view
  const int64_t v151 = 1;
  // pto: %hc_scale_2d_inline476__ssa_v0_view
  const int64_t v152 = 1;
  // pto: %hc_scale_2d_inline476__ssa_v0_view
  int64_t v153 = v66 * v69;
  // pto: %hc_scale_2d_inline476__ssa_v0_view
  int64_t v154 = v152 * v153;
  // pto: %hc_scale_2d_inline476__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v155 = pto::Shape<1, 1, 1, -1, -1>(v150, v151, v152, v66, v69);
  // pto: %hc_scale_2d_inline476__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v156 = pto::Stride<-1, -1, -1, -1, -1>(v151 * v154, v154, v153, v69, v66);
  // pto: %hc_scale_2d_inline476__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v157 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v10, v155, v156);
  // pto: %257
  int64_t v158 = (int64_t) ((uint64_t) v30 * (uint64_t) v94);
  // pto: %post_flat_inline541__ssa_v0_view
  const int64_t v159 = 1;
  // pto: %post_flat_inline541__ssa_v0_view
  const int64_t v160 = 1;
  // pto: %post_flat_inline541__ssa_v0_view
  const int64_t v161 = 1;
  // pto: %post_flat_inline541__ssa_v0_view
  int64_t v162 = v66 * v158;
  // pto: %post_flat_inline541__ssa_v0_view
  int64_t v163 = v161 * v162;
  // pto: %post_flat_inline541__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v164 = pto::Shape<1, 1, 1, -1, -1>(v159, v160, v161, v66, v158);
  // pto: %post_flat_inline541__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v165 = pto::Stride<-1, -1, -1, -1, -1>(v160 * v163, v163, v162, v158, v66);
  // pto: %post_flat_inline541__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v166 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v11, v164, v165);
  // pto: %mixed_view_inline442__ssa_v0_view
  const int64_t v167 = 1;
  // pto: %mixed_view_inline442__ssa_v0_view
  const int64_t v168 = 1;
  // pto: %mixed_view_inline442__ssa_v0_view
  const int64_t v169 = 1;
  // pto: %mixed_view_inline442__ssa_v0_view
  int64_t v170 = (int64_t) v30;
  // pto: %mixed_view_inline442__ssa_v0_view
  int64_t v171 = v170 * v71;
  // pto: %mixed_view_inline442__ssa_v0_view
  int64_t v172 = v169 * v171;
  // pto: %mixed_view_inline442__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v173 = pto::Shape<1, 1, 1, -1, -1>(v167, v168, v169, v170, v71);
  // pto: %mixed_view_inline442__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v174 = pto::Stride<-1, -1, -1, -1, -1>(v168 * v172, v172, v171, v71, v66);
  // pto: %mixed_view_inline442__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v175 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v12, v173, v174);
  const int32_t v176 = 0;
  auto v177 = TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>(v13, v176, v72);
  // pto: %worker_inline354__ssa_v0
  int64_t v178 = (int64_t) v31;
  // pto: %linear_workers_inline519__ssa_v0_idx
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  set_flag(PIPE_S, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
  for (int64_t v179 = v178; v179 < v14; v179 += (int64_t) v15) {
    // pto: %38, %39
    ;
    int64_t v180 = (int64_t) ((uint64_t) (v179 / v70) * (uint64_t) v73);
    // pto: %40
    ;
    int64_t v181 = v179 % v70;
    // pto: %mid_stream_rms_total_inline391__tile
    ;
    Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v182 = Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v67);
    // pto: %mid_stream_rms_total_inline391__tile
    ;
    uint64_t v183 = (uint64_t) v34;
    TASSIGN(v182, v183);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TEXPANDS(v182, v75);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    for (int64_t v184 = v76; v184 < v77; v184 += v66) {
      // pto: %41, %45, %44
      ;
      int64_t v185 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v181 * (uint64_t) v74) + (uint64_t) ((int64_t) (uint64_t) v184 * (uint64_t) v78));
      // pto: %47, %lane_inline440__ssa_v0, %46
      ;
      int64_t v186 = (int64_t) ((uint64_t) v180 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) v33) * (uint64_t) v67));
      // pto: %48
      ;
      int64_t v187 = (int64_t) ((uint64_t) v17 - (uint64_t) v186);
      // pto: %49
      ;
      int64_t v188 = v187 < v67 ? v187 : v67;
      // pto: %50
      ;
      int64_t v189 = v188 < v76 ? v76 : v188;
      // pto: %mid_stream_source_inline385__tile
      ;
      Tile<TileType::Vec, bfloat16_t, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v190 = Tile<TileType::Vec, bfloat16_t, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v189, v78);
      // pto: %mid_stream_source_inline385__tile
      ;
      uint64_t v191 = (uint64_t) v35;
      TASSIGN(v190, v191);
      // pto: %51
      ;
      int64_t v192 = v186 < v76 ? v76 : v186;
      // pto: %52
      ;
      int64_t v193 = v185 < v76 ? v76 : v185;
      // pto: %x_flat_inline459__ssa_v0_pview
      ;
      const int64_t v194 = 0;
      // pto: %x_flat_inline459__ssa_v0_pview
      ;
      __gm__ bfloat16_t* v195 = PTOAS__GLOBAL_TENSOR_DATA(v114);
      // pto: %x_flat_inline459__ssa_v0_pview
      ;
      const int64_t v196 = 1;
      // pto: %x_flat_inline459__ssa_v0_pview
      ;
      const int64_t v197 = 1;
      // pto: %x_flat_inline459__ssa_v0_pview
      ;
      const int64_t v198 = 1;
      // pto: %x_flat_inline459__ssa_v0_pview
      ;
      int64_t v199 = v189 * v68;
      // pto: %x_flat_inline459__ssa_v0_pview
      ;
      int64_t v200 = v198 * v199;
      // pto: %x_flat_inline459__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, -1> v201 = pto::Shape<1, 1, 1, -1, -1>(v196, v197, v198, v189, v78);
      // pto: %x_flat_inline459__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v202 = pto::Stride<-1, -1, -1, -1, -1>(v197 * v200, v200, v199, v68, v66);
      // pto: %x_flat_inline459__ssa_v0_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v203 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v195 + (v194 + v192 * v68 + v193 * v66), v201, v202);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v190, v203);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %mid_stream_widened_inline465__tile
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v204 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v189, v78);
      // pto: %mid_stream_widened_inline465__tile
      ;
      uint64_t v205 = (uint64_t) v36;
      TASSIGN(v204, v205);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      RoundMode v206 = RoundMode::CAST_ROUND;
      SaturationMode v207 = SaturationMode::ON;
      TCVT(v204, v190, v206, v207);
      v204.SetValidShape(v189, v78);
      // pto: %mid_stream_clean_inline400__tile
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v208 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v67, v78);
      // pto: %mid_stream_clean_inline400__tile
      ;
      uint64_t v209 = (uint64_t) v36;
      TASSIGN(v208, v209);
      pipe_barrier(PIPE_V);
      TFILLPAD<pto::TFillPadMode::InPlace>(v208, v204);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      v208.SetValidShape(v67, v78);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TPUSH<TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>, Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v177, v208);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      // pto: %0
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v210 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v67, v78);
      // pto: %0
      ;
      uint64_t v211 = (uint64_t) v36;
      TASSIGN(v210, v211);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      TMUL(v210, v208, v208);
      // pto: %tmp_tile
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v212 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v78);
      // pto: %tmp_tile
      ;
      uint64_t v213 = (uint64_t) v35;
      TASSIGN(v212, v213);
      // pto: %1
      ;
      Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v214 = Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v66);
      // pto: %1
      ;
      uint64_t v215 = (uint64_t) v37;
      TASSIGN(v214, v215);
      pipe_barrier(PIPE_V);
      TROWSUM(v214, v210, v212);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      // pto: %mid_stream_sq_inline439__tile
      ;
      Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v216 = Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v67);
      // pto: %mid_stream_sq_inline439__tile
      ;
      uint64_t v217 = (uint64_t) v37;
      TASSIGN(v216, v217);
      // pto: %2
      ;
      Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v218 = Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v67);
      // pto: %2
      ;
      uint64_t v219 = (uint64_t) v34;
      TASSIGN(v218, v219);
      pipe_barrier(PIPE_V);
      TADD(v218, v182, v216);
    };
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %42, %43, %54, %rms_lane_inline471__ssa_v0, %53
    ;
    int64_t v220 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v181 * (uint64_t) v16) + (uint64_t) v180) + (uint64_t) ((int64_t) (uint64_t) ((int64_t) v33) * (uint64_t) v67));
    // pto: %3
    ;
    Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v221 = Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v66);
    // pto: %3
    ;
    uint64_t v222 = (uint64_t) v34;
    TASSIGN(v221, v222);
    // pto: %55
    ;
    int64_t v223 = v220 < v76 ? v76 : v220;
    // pto: %rms_partials_inline548__iter_v1_pview
    ;
    const int64_t v224 = 0;
    // pto: %rms_partials_inline548__iter_v1_pview
    ;
    __gm__ float* v225 = PTOAS__GLOBAL_TENSOR_DATA(v105);
    // pto: %rms_partials_inline548__iter_v1_pview
    ;
    const int64_t v226 = 1;
    // pto: %rms_partials_inline548__iter_v1_pview
    ;
    const int64_t v227 = 1;
    // pto: %rms_partials_inline548__iter_v1_pview
    ;
    const int64_t v228 = 1;
    // pto: %rms_partials_inline548__iter_v1_pview
    ;
    int64_t v229 = v67 * v66;
    // pto: %rms_partials_inline548__iter_v1_pview
    ;
    int64_t v230 = v228 * v229;
    // pto: %rms_partials_inline548__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 24, 1> v231 = pto::Shape<1, 1, 1, 24, 1>(v226, v227, v228, v67, v66);
    // pto: %rms_partials_inline548__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v232 = pto::Stride<-1, -1, -1, -1, -1>(v227 * v230, v230, v229, v66, v97);
    // pto: %rms_partials_inline548__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 24, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v233 = GlobalTensor<float, pto::Shape<1, 1, 1, 24, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v225 + (v224 + v223 * v66 + v76 * v97), v231, v232);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    TSTORE(v233, v221);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  pipe_barrier(PIPE_ALL);
  dsb(DSB_DDR);
  SYNCALL<SyncCoreType::Mix>();
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  // pto: %56, %57, %tail_lane_inline371__ssa_v0
  int64_t v234 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v178 * (uint64_t) v77) + (uint64_t) ((int64_t) v33));
  // pto: %58
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  if (v234 < v18) {
    // pto: %60
    ;
    int64_t v235 = v234 < v20 ? v234 : v20;
    // pto: %59, %61
    ;
    int64_t v236 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v234 * (uint64_t) v19) + (uint64_t) v235);
    // pto: %63
    ;
    int64_t v237 = (int64_t) ((uint64_t) v20 - (uint64_t) v234);
    // pto: %64
    ;
    int64_t v238 = v237 < v76 ? v76 : v237;
    // pto: %65
    ;
    int64_t v239 = v238 < v66 ? v238 : v66;
    // pto: %62, %66
    ;
    int64_t v240 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v236 + (uint64_t) v19) + (uint64_t) v239);
    for (int64_t v241 = v236; v241 < v240; v241 += v77) {
      // pto: %scale2_inline363__tile
      ;
      float v242 = (v5)[v77];
      // pto: %67
      ;
      int64_t v243 = (int64_t) ((uint64_t) v240 - (uint64_t) v241);
      // pto: %68
      ;
      int64_t v244 = v243 < v77 ? v243 : v77;
      // pto: %69
      ;
      int64_t v245 = (int64_t) ((uint64_t) v244 * (uint64_t) v65);
      // pto: %c_parts_load_inline486__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v246 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v245);
      // pto: %c_parts_load_inline486__ssa_v0
      ;
      uint64_t v247 = (uint64_t) v36;
      TASSIGN(v246, v247);
      // pto: %70
      ;
      int64_t v248 = (int64_t) ((uint64_t) v241 * (uint64_t) v65);
      // pto: %71
      ;
      int64_t v249 = v248 < v76 ? v76 : v248;
      // pto: %parts_2d_inline445__ssa_v0_pview
      ;
      const int64_t v250 = 0;
      // pto: %parts_2d_inline445__ssa_v0_pview
      ;
      __gm__ float* v251 = PTOAS__GLOBAL_TENSOR_DATA(v123);
      // pto: %parts_2d_inline445__ssa_v0_pview
      ;
      const int64_t v252 = 1;
      // pto: %parts_2d_inline445__ssa_v0_pview
      ;
      const int64_t v253 = 1;
      // pto: %parts_2d_inline445__ssa_v0_pview
      ;
      const int64_t v254 = 1;
      // pto: %parts_2d_inline445__ssa_v0_pview
      ;
      int64_t v255 = v70 * v115;
      // pto: %parts_2d_inline445__ssa_v0_pview
      ;
      int64_t v256 = v254 * v255;
      // pto: %parts_2d_inline445__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 16, -1> v257 = pto::Shape<1, 1, 1, 16, -1>(v252, v253, v254, v70, v245);
      // pto: %parts_2d_inline445__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v258 = pto::Stride<-1, -1, -1, -1, -1>(v253 * v256, v256, v255, v115, v66);
      // pto: %parts_2d_inline445__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v259 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v251 + (v250 + v76 * v115 + v249 * v66), v257, v258);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      TLOAD(v246, v259);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %t__tmp_v94
      ;
      Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v260 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v70, v79);
      // pto: %t__tmp_v94
      ;
      uint64_t v261 = (uint64_t) v36;
      TASSIGN(v260, v261);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      TFILLPAD<pto::TFillPadMode::InPlace>(v260, v246);
      v260.SetValidShape(v70, v79);
      // pto: %c_parts_tmp_inline533__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v262 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v79);
      // pto: %c_parts_tmp_inline533__ssa_v0
      ;
      uint64_t v263 = (uint64_t) v38;
      TASSIGN(v262, v263);
      // pto: %t__tmp_v95
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v264 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v79);
      // pto: %t__tmp_v95
      ;
      uint64_t v265 = (uint64_t) v36;
      TASSIGN(v264, v265);
      // pto: %colsum_src_view
      ;
      Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, 16, 256, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v266;
      TRESHAPE(v266, v260);
      // pto: %colsum_tmp_view
      ;
      Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, 16, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v267;
      TRESHAPE(v267, v262);
      // pto: %colsum_dst_view
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v268;
      TRESHAPE(v268, v264);
      pipe_barrier(PIPE_V);
      const bool v269 = true;
      TCOLSUM(v268, v266, v267, v269);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
      // pto: %c_total_inline594__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v270 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v80, v65);
      // pto: %c_total_inline594__ssa_v0
      ;
      uint64_t v271 = (uint64_t) v36;
      TASSIGN(v270, v271);
      // pto: %c_sq_load_inline355__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v272 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v244);
      // pto: %c_sq_load_inline355__ssa_v0
      ;
      uint64_t v273 = (uint64_t) v39;
      TASSIGN(v272, v273);
      // pto: %73
      ;
      int64_t v274 = v241 < v76 ? v76 : v241;
      // pto: %rms_2d_inline408__ssa_v0_pview
      ;
      int64_t v275 = (int64_t) v29;
      // pto: %rms_2d_inline408__ssa_v0_pview
      ;
      const int64_t v276 = 0;
      // pto: %rms_2d_inline408__ssa_v0_pview
      ;
      __gm__ float* v277 = PTOAS__GLOBAL_TENSOR_DATA(v132);
      // pto: %rms_2d_inline408__ssa_v0_pview
      ;
      const int64_t v278 = 1;
      // pto: %rms_2d_inline408__ssa_v0_pview
      ;
      const int64_t v279 = 1;
      // pto: %rms_2d_inline408__ssa_v0_pview
      ;
      const int64_t v280 = 1;
      // pto: %rms_2d_inline408__ssa_v0_pview
      ;
      int64_t v281 = v70 * v275;
      // pto: %rms_2d_inline408__ssa_v0_pview
      ;
      int64_t v282 = v280 * v281;
      // pto: %rms_2d_inline408__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 16, -1> v283 = pto::Shape<1, 1, 1, 16, -1>(v278, v279, v280, v70, v244);
      // pto: %rms_2d_inline408__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v284 = pto::Stride<-1, -1, -1, -1, -1>(v279 * v282, v282, v281, v275, v66);
      // pto: %rms_2d_inline408__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v285 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v277 + (v276 + v76 * v275 + v274 * v66), v283, v284);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
      TLOAD(v272, v285);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %t__tmp_v96
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v286 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v70, v80);
      // pto: %t__tmp_v96
      ;
      uint64_t v287 = (uint64_t) v39;
      TASSIGN(v286, v287);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      TFILLPAD<pto::TFillPadMode::InPlace>(v286, v272);
      v286.SetValidShape(v70, v80);
      // pto: %c_sq_tmp_inline498__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v288 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v80);
      // pto: %c_sq_tmp_inline498__ssa_v0
      ;
      uint64_t v289 = (uint64_t) v40;
      TASSIGN(v288, v289);
      // pto: %c_rms_tmp_inline352__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v290 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v80);
      // pto: %c_rms_tmp_inline352__ssa_v0
      ;
      uint64_t v291 = (uint64_t) v41;
      TASSIGN(v290, v291);
      // pto: %t__tmp_v97
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v292 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v80);
      // pto: %t__tmp_v97
      ;
      uint64_t v293 = (uint64_t) v39;
      TASSIGN(v292, v293);
      // pto: %74
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, 16, 8, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v294;
      TRESHAPE(v294, v286);
      // pto: %75
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, 16, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v295;
      TRESHAPE(v295, v288);
      // pto: %76
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v296;
      TRESHAPE(v296, v292);
      pipe_barrier(PIPE_V);
      const bool v297 = true;
      TCOLSUM(v296, v294, v295, v297);
      // pto: %t__tmp_v98
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v298 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v80);
      // pto: %t__tmp_v98
      ;
      uint64_t v299 = (uint64_t) v39;
      TASSIGN(v298, v299);
      pipe_barrier(PIPE_V);
      TMULS(v298, v292, v81);
      // pto: %t__tmp_v99
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v300 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v80);
      // pto: %t__tmp_v99
      ;
      uint64_t v301 = (uint64_t) v39;
      TASSIGN(v300, v301);
      pipe_barrier(PIPE_V);
      TADDS(v300, v298, v82);
      // pto: %t__tmp_v100
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v302 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v80);
      // pto: %t__tmp_v100
      ;
      uint64_t v303 = (uint64_t) v42;
      TASSIGN(v302, v303);
      pipe_barrier(PIPE_V);
      TRSQRT(v302, v300, v290);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      // pto: %c_inv_inline366__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v304 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v80, v66);
      // pto: %c_inv_inline366__ssa_v0
      ;
      uint64_t v305 = (uint64_t) v42;
      TASSIGN(v304, v305);
      // pto: %comb_inv_inline510__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v306 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v80, v66);
      // pto: %comb_inv_inline510__ssa_v0
      ;
      uint64_t v307 = (uint64_t) v42;
      TASSIGN(v306, v307);
      // pto: %t__tmp_v101
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v308 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v80, v70);
      // pto: %t__tmp_v101
      ;
      uint64_t v309 = (uint64_t) v43;
      TASSIGN(v308, v309);
      // pto: %slice_view
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 16, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v310;
      // pto: %slice_view
      ;
      uint64_t v311 = (uint64_t) v43;
      TASSIGN(v310, v311);
      // pto: %raw_inline514__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v312 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v80, v70);
      // pto: %raw_inline514__ssa_v0
      ;
      uint64_t v313 = (uint64_t) v36;
      TASSIGN(v312, v313);
      TADDS(v312, v310, v75);
      // pto: %base_inline516__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v314 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v70);
      // pto: %base_inline516__ssa_v0
      ;
      uint64_t v315 = (uint64_t) v39;
      TASSIGN(v314, v315);
      // pto: %hc_base_2d_inline576__ssa_v0_pview
      ;
      __gm__ float* v316 = PTOAS__GLOBAL_TENSOR_DATA(v140);
      // pto: %hc_base_2d_inline576__ssa_v0_pview
      ;
      unsigned v317 = 8;
      // pto: %hc_base_2d_inline576__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 16> v318 = pto::Shape<1, 1, 1, 1, 16>();
      // pto: %hc_base_2d_inline576__ssa_v0_pview
      ;
      pto::Stride<24, 24, 24, 24, 1> v319 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %hc_base_2d_inline576__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v320 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v316 + v317, v318, v319);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      TLOAD(v314, v320);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      // pto: %t__tmp_v102
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v321 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v80, v70);
      // pto: %t__tmp_v102
      ;
      uint64_t v322 = (uint64_t) v44;
      TASSIGN(v321, v322);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v321, v312, v306);
      // pto: %t__tmp_v103
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v323 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v80, v70);
      // pto: %t__tmp_v103
      ;
      uint64_t v324 = (uint64_t) v44;
      TASSIGN(v323, v324);
      pipe_barrier(PIPE_V);
      TMULS(v323, v321, v242);
      // pto: %t__tmp_v104
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v325 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v80, v70);
      // pto: %t__tmp_v104
      ;
      uint64_t v326 = (uint64_t) v36;
      TASSIGN(v325, v326);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      TCOLEXPAND(v325, v314);
      // pto: %logits_inline518__ssa_v1
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v327 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v80, v70);
      // pto: %logits_inline518__ssa_v1
      ;
      uint64_t v328 = (uint64_t) v45;
      TASSIGN(v327, v328);
      pipe_barrier(PIPE_V);
      TADD(v327, v323, v325);
      // pto: %t__tmp_v105
      ;
      Tile<TileType::Vec, float, 2, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v329 = Tile<TileType::Vec, float, 2, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v77, v70);
      // pto: %t__tmp_v105
      ;
      uint64_t v330 = (uint64_t) v45;
      TASSIGN(v329, v330);
      // pto: %regular_logits_inline571__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v331 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %regular_logits_inline571__ssa_v0
      ;
      uint64_t v332 = (uint64_t) v45;
      TASSIGN(v331, v332);
      // pto: %ids_inline356__ci_tmp_v0
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v333 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v83);
      // pto: %ids_inline356__ci_tmp_v0
      ;
      uint64_t v334 = (uint64_t) v36;
      TASSIGN(v333, v334);
      // pto: %ids_inline356__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v335 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %ids_inline356__ssa_v1
      ;
      uint64_t v336 = (uint64_t) v46;
      TASSIGN(v335, v336);
      // pto: %ci_tmp_view
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v337;
      TRESHAPE(v337, v333);
      // pto: %ci_dst_view
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v338;
      TRESHAPE(v338, v335);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v338, v72, v337);
      // pto: %tmp_inline520__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v339 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %tmp_inline520__ssa_v1
      ;
      uint64_t v340 = (uint64_t) v36;
      TASSIGN(v339, v340);
      // pto: %ids16_inline441__ssa_v1
      ;
      Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v341 = Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v84);
      // pto: %ids16_inline441__ssa_v1
      ;
      uint64_t v342 = (uint64_t) v46;
      TASSIGN(v341, v342);
      // pto: %xor_tmp16_inline530__ssa_v1
      ;
      Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v343 = Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v84);
      // pto: %xor_tmp16_inline530__ssa_v1
      ;
      uint64_t v344 = (uint64_t) v47;
      TASSIGN(v343, v344);
      // pto: %t__tmp_v106
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v345 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %t__tmp_v106
      ;
      uint64_t v346 = (uint64_t) v48;
      TASSIGN(v345, v346);
      pipe_barrier(PIPE_V);
      TEXPANDS(v345, v85);
      // pto: %ri1_mask_inline481__ssa_v1
      ;
      Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v347 = Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v84);
      // pto: %ri1_mask_inline481__ssa_v1
      ;
      uint64_t v348 = (uint64_t) v48;
      TASSIGN(v347, v348);
      // pto: %t__tmp_v107
      ;
      Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v349 = Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v84);
      // pto: %t__tmp_v107
      ;
      uint64_t v350 = (uint64_t) v49;
      TASSIGN(v349, v350);
      pipe_barrier(PIPE_V);
      TXOR(v349, v341, v347, v343);
      // pto: %ri1_inline436__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v351 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %ri1_inline436__ssa_v1
      ;
      uint64_t v352 = (uint64_t) v49;
      TASSIGN(v351, v352);
      // pto: %t__tmp_v108
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v353 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %t__tmp_v108
      ;
      uint64_t v354 = (uint64_t) v48;
      TASSIGN(v353, v354);
      pipe_barrier(PIPE_V);
      TEXPANDS(v353, v86);
      // pto: %ri2_mask_inline568__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v355 = Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v84);
      // pto: %ri2_mask_inline568__ssa_v0
      ;
      uint64_t v356 = (uint64_t) v48;
      TASSIGN(v355, v356);
      // pto: %t__tmp_v109
      ;
      Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v357 = Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v84);
      // pto: %t__tmp_v109
      ;
      uint64_t v358 = (uint64_t) v44;
      TASSIGN(v357, v358);
      pipe_barrier(PIPE_V);
      TXOR(v357, v341, v355, v343);
      // pto: %ri2_inline572__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v359 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %ri2_inline572__ssa_v0
      ;
      uint64_t v360 = (uint64_t) v44;
      TASSIGN(v359, v360);
      // pto: %t__tmp_v110
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v361 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %t__tmp_v110
      ;
      uint64_t v362 = (uint64_t) v48;
      TASSIGN(v361, v362);
      pipe_barrier(PIPE_V);
      TEXPANDS(v361, v87);
      // pto: %ci1_mask_inline574__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v363 = Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v84);
      // pto: %ci1_mask_inline574__ssa_v0
      ;
      uint64_t v364 = (uint64_t) v48;
      TASSIGN(v363, v364);
      // pto: %t__tmp_v111
      ;
      Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v365 = Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v84);
      // pto: %t__tmp_v111
      ;
      uint64_t v366 = (uint64_t) v50;
      TASSIGN(v365, v366);
      pipe_barrier(PIPE_V);
      TXOR(v365, v341, v363, v343);
      // pto: %ci1_inline575__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v367 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %ci1_inline575__ssa_v0
      ;
      uint64_t v368 = (uint64_t) v50;
      TASSIGN(v367, v368);
      // pto: %t__tmp_v112
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v369 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %t__tmp_v112
      ;
      uint64_t v370 = (uint64_t) v48;
      TASSIGN(v369, v370);
      pipe_barrier(PIPE_V);
      TEXPANDS(v369, v88);
      // pto: %ci2_mask_inline580__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v371 = Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v84);
      // pto: %ci2_mask_inline580__ssa_v0
      ;
      uint64_t v372 = (uint64_t) v48;
      TASSIGN(v371, v372);
      // pto: %t__tmp_v113
      ;
      Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v373 = Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v84);
      // pto: %t__tmp_v113
      ;
      uint64_t v374 = (uint64_t) v51;
      TASSIGN(v373, v374);
      pipe_barrier(PIPE_V);
      TXOR(v373, v341, v371, v343);
      // pto: %ci2_inline581__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v375 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %ci2_inline581__ssa_v0
      ;
      uint64_t v376 = (uint64_t) v51;
      TASSIGN(v375, v376);
      // pto: %t__tmp_v114
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v377 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %t__tmp_v114
      ;
      uint64_t v378 = (uint64_t) v47;
      TASSIGN(v377, v378);
      pipe_barrier(PIPE_V);
      TGATHER(v377, v331, v351, v339);
      // pto: %max2_inline586__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v379 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %max2_inline586__ssa_v1
      ;
      uint64_t v380 = (uint64_t) v47;
      TASSIGN(v379, v380);
      pipe_barrier(PIPE_V);
      TMAX(v379, v331, v377);
      // pto: %t__tmp_v115
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v381 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %t__tmp_v115
      ;
      uint64_t v382 = (uint64_t) v48;
      TASSIGN(v381, v382);
      pipe_barrier(PIPE_V);
      TGATHER(v381, v379, v359, v339);
      // pto: %max4_inline529__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v383 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %max4_inline529__ssa_v1
      ;
      uint64_t v384 = (uint64_t) v47;
      TASSIGN(v383, v384);
      pipe_barrier(PIPE_V);
      TMAX(v383, v379, v381);
      // pto: %t__tmp_v116
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v385 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %t__tmp_v116
      ;
      uint64_t v386 = (uint64_t) v47;
      TASSIGN(v385, v386);
      pipe_barrier(PIPE_V);
      TSUB(v385, v331, v383);
      // pto: %exps_inline553__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v387 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %exps_inline553__ssa_v1
      ;
      uint64_t v388 = (uint64_t) v47;
      TASSIGN(v387, v388);
      pipe_barrier(PIPE_V);
      TEXP(v387, v385);
      // pto: %t__tmp_v117
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v389 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %t__tmp_v117
      ;
      uint64_t v390 = (uint64_t) v48;
      TASSIGN(v389, v390);
      pipe_barrier(PIPE_V);
      TGATHER(v389, v387, v351, v339);
      // pto: %sum2_inline546__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v391 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %sum2_inline546__ssa_v1
      ;
      uint64_t v392 = (uint64_t) v48;
      TASSIGN(v391, v392);
      pipe_barrier(PIPE_V);
      TADD(v391, v387, v389);
      // pto: %t__tmp_v118
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v393 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %t__tmp_v118
      ;
      uint64_t v394 = (uint64_t) v45;
      TASSIGN(v393, v394);
      pipe_barrier(PIPE_V);
      TGATHER(v393, v391, v359, v339);
      // pto: %sum4_inline549__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v395 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %sum4_inline549__ssa_v1
      ;
      uint64_t v396 = (uint64_t) v48;
      TASSIGN(v395, v396);
      pipe_barrier(PIPE_V);
      TADD(v395, v391, v393);
      // pto: %t__tmp_v119
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v397 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %t__tmp_v119
      ;
      uint64_t v398 = (uint64_t) v47;
      TASSIGN(v397, v398);
      pipe_barrier(PIPE_V);
      TDIV(v397, v387, v395);
      // pto: %probs_inline551__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v399 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %probs_inline551__ssa_v1
      ;
      uint64_t v400 = (uint64_t) v47;
      TASSIGN(v399, v400);
      pipe_barrier(PIPE_V);
      TADDS(v399, v397, v82);
      // pto: %t__tmp_v120
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v401 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %t__tmp_v120
      ;
      uint64_t v402 = (uint64_t) v48;
      TASSIGN(v401, v402);
      pipe_barrier(PIPE_V);
      TGATHER(v401, v399, v367, v339);
      // pto: %col2_inline415__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v403 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %col2_inline415__ssa_v4
      ;
      uint64_t v404 = (uint64_t) v48;
      TASSIGN(v403, v404);
      pipe_barrier(PIPE_V);
      TADD(v403, v399, v401);
      // pto: %t__tmp_v121
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v405 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %t__tmp_v121
      ;
      uint64_t v406 = (uint64_t) v45;
      TASSIGN(v405, v406);
      pipe_barrier(PIPE_V);
      TGATHER(v405, v403, v375, v339);
      // pto: %t__tmp_v122
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v407 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %t__tmp_v122
      ;
      uint64_t v408 = (uint64_t) v48;
      TASSIGN(v407, v408);
      pipe_barrier(PIPE_V);
      TADD(v407, v403, v405);
      // pto: %col4_inline559__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v409 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %col4_inline559__ssa_v4
      ;
      uint64_t v410 = (uint64_t) v48;
      TASSIGN(v409, v410);
      pipe_barrier(PIPE_V);
      TADDS(v409, v407, v82);
      // pto: %cur_inline472__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v411 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %cur_inline472__ssa_v4
      ;
      uint64_t v412 = (uint64_t) v47;
      TASSIGN(v411, v412);
      pipe_barrier(PIPE_V);
      TDIV(v411, v399, v409);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      for (int64_t v413 = v76; v413 < v89; v413 += v66) {
        // pto: %t__tmp_v123
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v414 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
        // pto: %t__tmp_v123
        ;
        uint64_t v415 = (uint64_t) v48;
        TASSIGN(v414, v415);
        pipe_barrier(PIPE_V);
        TGATHER(v414, v411, v351, v339);
        // pto: %row2_inline579__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v416 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
        // pto: %row2_inline579__ssa_v1
        ;
        uint64_t v417 = (uint64_t) v48;
        TASSIGN(v416, v417);
        pipe_barrier(PIPE_V);
        TADD(v416, v411, v414);
        // pto: %t__tmp_v124
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v418 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
        // pto: %t__tmp_v124
        ;
        uint64_t v419 = (uint64_t) v45;
        TASSIGN(v418, v419);
        pipe_barrier(PIPE_V);
        TGATHER(v418, v416, v359, v339);
        // pto: %t__tmp_v125
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v420 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
        // pto: %t__tmp_v125
        ;
        uint64_t v421 = (uint64_t) v48;
        TASSIGN(v420, v421);
        pipe_barrier(PIPE_V);
        TADD(v420, v416, v418);
        // pto: %row4_inline639__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v422 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
        // pto: %row4_inline639__ssa_v1
        ;
        uint64_t v423 = (uint64_t) v48;
        TASSIGN(v422, v423);
        pipe_barrier(PIPE_V);
        TADDS(v422, v420, v82);
        // pto: %norm_inline569__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v424 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
        // pto: %norm_inline569__ssa_v1
        ;
        uint64_t v425 = (uint64_t) v48;
        TASSIGN(v424, v425);
        pipe_barrier(PIPE_V);
        TDIV(v424, v411, v422);
        // pto: %t__tmp_v126
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v426 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
        // pto: %t__tmp_v126
        ;
        uint64_t v427 = (uint64_t) v45;
        TASSIGN(v426, v427);
        pipe_barrier(PIPE_V);
        TGATHER(v426, v424, v367, v339);
        // pto: %col2_inline415__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v428 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
        // pto: %col2_inline415__ssa_v7
        ;
        uint64_t v429 = (uint64_t) v45;
        TASSIGN(v428, v429);
        pipe_barrier(PIPE_V);
        TADD(v428, v424, v426);
        // pto: %t__tmp_v127
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v430 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
        // pto: %t__tmp_v127
        ;
        uint64_t v431 = (uint64_t) v52;
        TASSIGN(v430, v431);
        pipe_barrier(PIPE_V);
        TGATHER(v430, v428, v375, v339);
        // pto: %t__tmp_v128
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v432 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
        // pto: %t__tmp_v128
        ;
        uint64_t v433 = (uint64_t) v45;
        TASSIGN(v432, v433);
        pipe_barrier(PIPE_V);
        TADD(v432, v428, v430);
        // pto: %col4_inline559__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v434 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
        // pto: %col4_inline559__ssa_v7
        ;
        uint64_t v435 = (uint64_t) v45;
        TASSIGN(v434, v435);
        pipe_barrier(PIPE_V);
        TADDS(v434, v432, v82);
        // pto: %cur_inline472__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v436 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
        // pto: %cur_inline472__ssa_v7
        ;
        uint64_t v437 = (uint64_t) v47;
        TASSIGN(v436, v437);
        pipe_barrier(PIPE_V);
        TDIV(v436, v424, v434);
      };
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      // pto: %t__tmp_v129
      ;
      Tile<TileType::Vec, float, 2, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v438 = Tile<TileType::Vec, float, 2, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v77, v70);
      // pto: %t__tmp_v129
      ;
      uint64_t v439 = (uint64_t) v47;
      TASSIGN(v438, v439);
      v438.SetValidShape(v244, v70);
      // pto: %comb_inline367__ssa_v0_pview
      ;
      const int64_t v440 = 0;
      // pto: %comb_inline367__ssa_v0_pview
      ;
      __gm__ float* v441 = PTOAS__GLOBAL_TENSOR_DATA(v149);
      // pto: %comb_inline367__ssa_v0_pview
      ;
      const int64_t v442 = 1;
      // pto: %comb_inline367__ssa_v0_pview
      ;
      const int64_t v443 = 1;
      // pto: %comb_inline367__ssa_v0_pview
      ;
      const int64_t v444 = 1;
      // pto: %comb_inline367__ssa_v0_pview
      ;
      int64_t v445 = v244 * v70;
      // pto: %comb_inline367__ssa_v0_pview
      ;
      int64_t v446 = v444 * v445;
      // pto: %comb_inline367__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, 16> v447 = pto::Shape<1, 1, 1, -1, 16>(v442, v443, v444, v244, v70);
      // pto: %comb_inline367__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v448 = pto::Stride<-1, -1, -1, -1, -1>(v443 * v446, v446, v445, v70, v66);
      // pto: %comb_inline367__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v449 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v441 + (v440 + v274 * v70 + v76 * v66), v447, v448);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      TSTORE(v449, v438);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    };
  } else {
    // pto: %79
    ;
    int64_t v450 = (int64_t) ((uint64_t) v234 - (uint64_t) v18);
    // pto: %81
    ;
    int64_t v451 = v450 < v22 ? v450 : v22;
    // pto: %80, %82, %83
    ;
    int64_t v452 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v450 * (uint64_t) v21) + (uint64_t) v451) * (uint64_t) v77);
    // pto: %84
    ;
    int64_t v453 = (int64_t) ((uint64_t) v22 - (uint64_t) v450);
    // pto: %85
    ;
    int64_t v454 = v453 < v76 ? v76 : v453;
    // pto: %86
    ;
    int64_t v455 = v454 < v66 ? v454 : v66;
    // pto: %89, %87, %88
    ;
    int64_t v456 = (int64_t) ((uint64_t) v452 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v21 + (uint64_t) v455) * (uint64_t) v77));
    // pto: %90
    ;
    // pto: %91
    ;
    // pto: %d_begin_inline532__phi_v2
    ;
    int64_t v457;
    // pto: %d_end_inline626__phi_v2
    ;
    int64_t v458;
    // pto: %work_begin_inline420__phi_v3
    ;
    int64_t v459;
    // pto: %work_end_inline428__phi_v3
    ;
    int64_t v460;
    if (v24 < v25) {
      // pto: %92
      ;
      // pto: %d_worker_inline398__phi_v2
      ;
      int64_t v461;
      // pto: %d_workers_inline630__phi_v2
      ;
      int64_t v462;
      // pto: %pair_id_inline390__phi_v2
      ;
      int64_t v463;
      if (v450 < v26) {
        // pto: %93
        ;
        int64_t v464 = (int64_t) ((uint64_t) v27 + (uint64_t) v66);
        // pto: %94
        ;
        // pto: %96
        ;
        v461 = v450 % v464;
        v462 = v464;
        v463 = v450 / v464;
      } else {
        // pto: %98
        ;
        int64_t v465 = (int64_t) ((uint64_t) v450 - (uint64_t) v26);
        // pto: %100, %99
        ;
        // pto: %101
        ;
        v461 = v465 % v27;
        v462 = v27;
        v463 = (int64_t) ((uint64_t) v28 + (uint64_t) (v465 / v27));
      };
      // pto: %102
      ;
      uint64_t v466 = (uint64_t) v463;
      // pto: %102
      ;
      int64_t v467 = (int64_t) (v466 * (uint64_t) v77);
      // pto: %103
      ;
      int64_t v468 = v23 < v467 ? v23 : v467;
      // pto: %104
      ;
      int64_t v469 = (int64_t) ((uint64_t) v468 + (uint64_t) v77);
      // pto: %105
      ;
      // pto: %106
      ;
      int64_t v470 = v80 / v462;
      // pto: %107
      ;
      int64_t v471 = v80 % v462;
      // pto: %108
      ;
      uint64_t v472 = (uint64_t) v461;
      // pto: %109
      ;
      bool v473 = v461 < v471;
      // pto: %109
      ;
      int64_t v474 = v473 ? v461 : v471;
      // pto: %108, %110
      ;
      int64_t v475 = (int64_t) ((uint64_t) ((int64_t) v472 * (uint64_t) v470) + (uint64_t) v474);
      // pto: %112
      ;
      uint64_t v476 = (uint64_t) v461;
      // pto: %112
      ;
      int64_t v477 = (int64_t) ((uint64_t) v471 - v476);
      // pto: %113
      ;
      int64_t v478 = v477 < v76 ? v76 : v477;
      // pto: %114
      ;
      int64_t v479 = v478 < v66 ? v478 : v66;
      // pto: %111, %115
      ;
      v457 = v475;
      v458 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v475 + (uint64_t) v470) + (uint64_t) v479);
      v459 = v468;
      v460 = v23 < v469 ? v23 : v469;
    } else {
      v457 = v76;
      v458 = v80;
      v459 = v452;
      v460 = v23 < v456 ? v23 : v456;
    };
    // pto: %116
    ;
    bool v480 = v25 <= v450;
    // pto: %work_begin_inline420__phi_v5
    ;
    int64_t v481 = v480 ? v76 : v459;
    // pto: %work_end_inline428__phi_v5
    ;
    int64_t v482 = v480 ? v76 : v460;
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    for (int64_t v483 = v481; v483 < v482; v483 += v77) {
      // pto: %117
      ;
      int64_t v484 = (int64_t) ((uint64_t) v482 - (uint64_t) v483);
      // pto: %118
      ;
      int64_t v485 = v484 < v77 ? v484 : v77;
      // pto: %119
      ;
      int64_t v486 = (int64_t) ((uint64_t) v485 * (uint64_t) v65);
      // pto: %m_parts_load_inline537__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v487 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v486);
      // pto: %m_parts_load_inline537__ssa_v0
      ;
      uint64_t v488 = (uint64_t) v36;
      TASSIGN(v487, v488);
      // pto: %120
      ;
      int64_t v489 = (int64_t) ((uint64_t) v483 * (uint64_t) v65);
      // pto: %121
      ;
      int64_t v490 = v489 < v76 ? v76 : v489;
      // pto: %123
      ;
      const int64_t v491 = 0;
      // pto: %123
      ;
      __gm__ float* v492 = PTOAS__GLOBAL_TENSOR_DATA(v123);
      // pto: %123
      ;
      const int64_t v493 = 1;
      // pto: %123
      ;
      const int64_t v494 = 1;
      // pto: %123
      ;
      const int64_t v495 = 1;
      // pto: %123
      ;
      int64_t v496 = v70 * v115;
      // pto: %123
      ;
      int64_t v497 = v495 * v496;
      // pto: %123
      ;
      pto::Shape<1, 1, 1, 16, -1> v498 = pto::Shape<1, 1, 1, 16, -1>(v493, v494, v495, v70, v486);
      // pto: %123
      ;
      pto::Stride<-1, -1, -1, -1, -1> v499 = pto::Stride<-1, -1, -1, -1, -1>(v494 * v497, v497, v496, v115, v66);
      // pto: %123
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v500 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v492 + (v491 + v76 * v115 + v490 * v66), v498, v499);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      TLOAD(v487, v500);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      // pto: %t__tmp_v130
      ;
      Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v501 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v70, v79);
      // pto: %t__tmp_v130
      ;
      uint64_t v502 = (uint64_t) v36;
      TASSIGN(v501, v502);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      TFILLPAD<pto::TFillPadMode::InPlace>(v501, v487);
      v501.SetValidShape(v70, v79);
      // pto: %m_parts_tmp_inline550__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v503 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v79);
      // pto: %m_parts_tmp_inline550__ssa_v0
      ;
      uint64_t v504 = (uint64_t) v38;
      TASSIGN(v503, v504);
      // pto: %t__tmp_v131
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v505 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v79);
      // pto: %t__tmp_v131
      ;
      uint64_t v506 = (uint64_t) v36;
      TASSIGN(v505, v506);
      // pto: %124
      ;
      Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, 16, 256, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v507;
      TRESHAPE(v507, v501);
      // pto: %125
      ;
      Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, 16, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v508;
      TRESHAPE(v508, v503);
      // pto: %126
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v509;
      TRESHAPE(v509, v505);
      pipe_barrier(PIPE_V);
      const bool v510 = true;
      TCOLSUM(v509, v507, v508, v510);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      // pto: %m_total_inline618__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v511 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v80, v65);
      // pto: %m_total_inline618__ssa_v0
      ;
      uint64_t v512 = (uint64_t) v36;
      TASSIGN(v511, v512);
      // pto: %m_sq_load_inline633__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v513 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v485);
      // pto: %m_sq_load_inline633__ssa_v0
      ;
      uint64_t v514 = (uint64_t) v39;
      TASSIGN(v513, v514);
      // pto: %127
      ;
      int64_t v515 = v483 < v76 ? v76 : v483;
      // pto: %128
      ;
      int64_t v516 = (int64_t) v29;
      // pto: %128
      ;
      const int64_t v517 = 0;
      // pto: %128
      ;
      __gm__ float* v518 = PTOAS__GLOBAL_TENSOR_DATA(v132);
      // pto: %128
      ;
      const int64_t v519 = 1;
      // pto: %128
      ;
      const int64_t v520 = 1;
      // pto: %128
      ;
      const int64_t v521 = 1;
      // pto: %128
      ;
      int64_t v522 = v70 * v516;
      // pto: %128
      ;
      int64_t v523 = v521 * v522;
      // pto: %128
      ;
      pto::Shape<1, 1, 1, 16, -1> v524 = pto::Shape<1, 1, 1, 16, -1>(v519, v520, v521, v70, v485);
      // pto: %128
      ;
      pto::Stride<-1, -1, -1, -1, -1> v525 = pto::Stride<-1, -1, -1, -1, -1>(v520 * v523, v523, v522, v516, v66);
      // pto: %128
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v526 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v518 + (v517 + v76 * v516 + v515 * v66), v524, v525);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      TLOAD(v513, v526);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      // pto: %t__tmp_v132
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v527 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v70, v80);
      // pto: %t__tmp_v132
      ;
      uint64_t v528 = (uint64_t) v39;
      TASSIGN(v527, v528);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      TFILLPAD<pto::TFillPadMode::InPlace>(v527, v513);
      v527.SetValidShape(v70, v80);
      // pto: %m_sq_tmp_inline583__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v529 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v80);
      // pto: %m_sq_tmp_inline583__ssa_v0
      ;
      uint64_t v530 = (uint64_t) v40;
      TASSIGN(v529, v530);
      // pto: %m_rms_tmp_inline595__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v531 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v80);
      // pto: %m_rms_tmp_inline595__ssa_v0
      ;
      uint64_t v532 = (uint64_t) v41;
      TASSIGN(v531, v532);
      // pto: %t__tmp_v133
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v533 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v80);
      // pto: %t__tmp_v133
      ;
      uint64_t v534 = (uint64_t) v39;
      TASSIGN(v533, v534);
      // pto: %129
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, 16, 8, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v535;
      TRESHAPE(v535, v527);
      // pto: %130
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, 16, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v536;
      TRESHAPE(v536, v529);
      // pto: %131
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v537;
      TRESHAPE(v537, v533);
      pipe_barrier(PIPE_V);
      const bool v538 = true;
      TCOLSUM(v537, v535, v536, v538);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      // pto: %t__tmp_v134
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v539 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v80);
      // pto: %t__tmp_v134
      ;
      uint64_t v540 = (uint64_t) v39;
      TASSIGN(v539, v540);
      pipe_barrier(PIPE_V);
      TMULS(v539, v533, v81);
      // pto: %t__tmp_v135
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v541 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v80);
      // pto: %t__tmp_v135
      ;
      uint64_t v542 = (uint64_t) v39;
      TASSIGN(v541, v542);
      pipe_barrier(PIPE_V);
      TADDS(v541, v539, v82);
      // pto: %t__tmp_v136
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v543 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v80);
      // pto: %t__tmp_v136
      ;
      uint64_t v544 = (uint64_t) v53;
      TASSIGN(v543, v544);
      pipe_barrier(PIPE_V);
      TRSQRT(v543, v541, v531);
      // pto: %m_inv_inline450__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v545 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v80, v66);
      // pto: %m_inv_inline450__ssa_v0
      ;
      uint64_t v546 = (uint64_t) v53;
      TASSIGN(v545, v546);
      // pto: %gate_base_inline637__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v547 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v80);
      // pto: %gate_base_inline637__ssa_v0
      ;
      uint64_t v548 = (uint64_t) v54;
      TASSIGN(v547, v548);
      // pto: %132
      ;
      __gm__ float* v549 = PTOAS__GLOBAL_TENSOR_DATA(v140);
      // pto: %132
      ;
      pto::Shape<1, 1, 1, 1, 8> v550 = pto::Shape<1, 1, 1, 1, 8>();
      // pto: %132
      ;
      pto::Stride<24, 24, 24, 24, 1> v551 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %132
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v552 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v549, v550, v551);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      TLOAD(v547, v552);
      // pto: %gate_scales_inline638__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v553 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v69);
      // pto: %gate_scales_inline638__ssa_v0
      ;
      uint64_t v554 = (uint64_t) v55;
      TASSIGN(v553, v554);
      // pto: %hc_scale_2d_inline476__ssa_v0_pview
      ;
      __gm__ float* v555 = PTOAS__GLOBAL_TENSOR_DATA(v157);
      // pto: %hc_scale_2d_inline476__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 3> v556 = pto::Shape<1, 1, 1, 1, 3>();
      // pto: %hc_scale_2d_inline476__ssa_v0_pview
      ;
      pto::Stride<3, 3, 3, 3, 1> v557 = pto::Stride<3, 3, 3, 3, 1>();
      // pto: %hc_scale_2d_inline476__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND> v558 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND>(v555, v556, v557);
      TLOAD(v553, v558);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      // pto: %t__ci_tmp_v1
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v559 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v83);
      // pto: %t__ci_tmp_v1
      ;
      uint64_t v560 = (uint64_t) v39;
      TASSIGN(v559, v560);
      // pto: %t__tmp_v137
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v561 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v80);
      // pto: %t__tmp_v137
      ;
      uint64_t v562 = (uint64_t) v56;
      TASSIGN(v561, v562);
      // pto: %133
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v563;
      TRESHAPE(v563, v559);
      // pto: %134
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v564;
      TRESHAPE(v564, v561);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v564, v72, v563);
      // pto: %gate_ids_inline410__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v565 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v80);
      // pto: %gate_ids_inline410__ssa_v0
      ;
      uint64_t v566 = (uint64_t) v39;
      TASSIGN(v565, v566);
      pipe_barrier(PIPE_V);
      RoundMode v567 = RoundMode::CAST_ROUND;
      SaturationMode v568 = SaturationMode::ON;
      TCVT(v565, v561, v567, v568);
      // pto: %t__tmp_v138
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v569 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v80);
      // pto: %t__tmp_v138
      ;
      uint64_t v570 = (uint64_t) v39;
      TASSIGN(v569, v570);
      pipe_barrier(PIPE_V);
      TMULS(v569, v565, v90);
      // pto: %gate_scale_ids_inline612__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v571 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v80);
      // pto: %gate_scale_ids_inline612__ssa_v0
      ;
      uint64_t v572 = (uint64_t) v39;
      TASSIGN(v571, v572);
      pipe_barrier(PIPE_V);
      RoundMode v573 = RoundMode::CAST_FLOOR;
      SaturationMode v574 = SaturationMode::ON;
      TCVT(v571, v569, v573, v574);
      // pto: %gate_tmp_inline627__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v575 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v80);
      // pto: %gate_tmp_inline627__ssa_v0
      ;
      uint64_t v576 = (uint64_t) v57;
      TASSIGN(v575, v576);
      // pto: %gate_scale_vec_inline609__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v577 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v80);
      // pto: %gate_scale_vec_inline609__ssa_v0
      ;
      uint64_t v578 = (uint64_t) v42;
      TASSIGN(v577, v578);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      TGATHER(v577, v553, v571, v575);
      // pto: %t__tmp_v139
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v579 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v80, v80);
      // pto: %t__tmp_v139
      ;
      uint64_t v580 = (uint64_t) v36;
      TASSIGN(v579, v580);
      // pto: %135
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v581;
      // pto: %135
      ;
      uint64_t v582 = (uint64_t) v36;
      TASSIGN(v581, v582);
      // pto: %gate_raw_inline640__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v583 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v80, v80);
      // pto: %gate_raw_inline640__ssa_v0
      ;
      uint64_t v584 = (uint64_t) v36;
      TASSIGN(v583, v584);
      TADDS(v583, v581, v75);
      // pto: %gate_norm_inline641__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v585 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v80, v80);
      // pto: %gate_norm_inline641__ssa_v0
      ;
      uint64_t v586 = (uint64_t) v36;
      TASSIGN(v585, v586);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v585, v583, v545);
      // pto: %gate_scaled_inline602__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v587 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v80, v80);
      // pto: %gate_scaled_inline602__ssa_v0
      ;
      uint64_t v588 = (uint64_t) v36;
      TASSIGN(v587, v588);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v587, v585, v577);
      // pto: %t__tmp_v140
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v589 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v80, v80);
      // pto: %t__tmp_v140
      ;
      uint64_t v590 = (uint64_t) v48;
      TASSIGN(v589, v590);
      TCOLEXPAND(v589, v547);
      // pto: %gate_logits_inline643__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v591 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v80, v80);
      // pto: %gate_logits_inline643__ssa_v0
      ;
      uint64_t v592 = (uint64_t) v36;
      TASSIGN(v591, v592);
      pipe_barrier(PIPE_V);
      TADD(v591, v587, v589);
      // pto: %t__tmp_v141
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v593 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v80, v80);
      // pto: %t__tmp_v141
      ;
      uint64_t v594 = (uint64_t) v36;
      TASSIGN(v593, v594);
      pipe_barrier(PIPE_V);
      TNEG(v593, v591);
      // pto: %t__tmp_v142
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v595 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v80, v80);
      // pto: %t__tmp_v142
      ;
      uint64_t v596 = (uint64_t) v36;
      TASSIGN(v595, v596);
      pipe_barrier(PIPE_V);
      TEXP(v595, v593);
      // pto: %t__tmp_v143
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v597 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v80, v80);
      // pto: %t__tmp_v143
      ;
      uint64_t v598 = (uint64_t) v36;
      TASSIGN(v597, v598);
      pipe_barrier(PIPE_V);
      TADDS(v597, v595, v91);
      // pto: %gate_sigmoid_inline351__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v599 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v80, v80);
      // pto: %gate_sigmoid_inline351__ssa_v0
      ;
      uint64_t v600 = (uint64_t) v58;
      TASSIGN(v599, v600);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
      TRECIP(v599, v597);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      // pto: %t__ci_tmp_v2
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v601 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v83);
      // pto: %t__ci_tmp_v2
      ;
      uint64_t v602 = (uint64_t) v36;
      TASSIGN(v601, v602);
      // pto: %t__tmp_v144
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v603 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %t__tmp_v144
      ;
      uint64_t v604 = (uint64_t) v45;
      TASSIGN(v603, v604);
      // pto: %136
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v605;
      TRESHAPE(v605, v601);
      // pto: %137
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v606;
      TRESHAPE(v606, v603);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v606, v72, v605);
      // pto: %post_ids_float_inline604__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v607 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %post_ids_float_inline604__ssa_v0
      ;
      uint64_t v608 = (uint64_t) v36;
      TASSIGN(v607, v608);
      pipe_barrier(PIPE_V);
      RoundMode v609 = RoundMode::CAST_ROUND;
      SaturationMode v610 = SaturationMode::ON;
      TCVT(v607, v603, v609, v610);
      // pto: %t__tmp_v145
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v611 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %t__tmp_v145
      ;
      uint64_t v612 = (uint64_t) v47;
      TASSIGN(v611, v612);
      pipe_barrier(PIPE_V);
      TMULS(v611, v607, v90);
      // pto: %t__tmp_v146
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v613 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %t__tmp_v146
      ;
      uint64_t v614 = (uint64_t) v47;
      TASSIGN(v613, v614);
      pipe_barrier(PIPE_V);
      RoundMode v615 = RoundMode::CAST_FLOOR;
      SaturationMode v616 = SaturationMode::ON;
      TCVT(v613, v611, v615, v616);
      // pto: %post_rows_inline349__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v617 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %post_rows_inline349__ssa_v0
      ;
      uint64_t v618 = (uint64_t) v47;
      TASSIGN(v617, v618);
      pipe_barrier(PIPE_V);
      RoundMode v619 = RoundMode::CAST_ROUND;
      SaturationMode v620 = SaturationMode::ON;
      TCVT(v617, v613, v619, v620);
      // pto: %t__tmp_v147
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v621 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %t__tmp_v147
      ;
      uint64_t v622 = (uint64_t) v47;
      TASSIGN(v621, v622);
      pipe_barrier(PIPE_V);
      TMULS(v621, v617, v92);
      // pto: %t__tmp_v148
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v623 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %t__tmp_v148
      ;
      uint64_t v624 = (uint64_t) v36;
      TASSIGN(v623, v624);
      pipe_barrier(PIPE_V);
      TADD(v623, v607, v621);
      // pto: %t__tmp_v149
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v625 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %t__tmp_v149
      ;
      uint64_t v626 = (uint64_t) v36;
      TASSIGN(v625, v626);
      pipe_barrier(PIPE_V);
      TADDS(v625, v623, v92);
      // pto: %post_ids_inline493__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v627 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %post_ids_inline493__ssa_v0
      ;
      uint64_t v628 = (uint64_t) v36;
      TASSIGN(v627, v628);
      pipe_barrier(PIPE_V);
      RoundMode v629 = RoundMode::CAST_ROUND;
      SaturationMode v630 = SaturationMode::ON;
      TCVT(v627, v625, v629, v630);
      // pto: %post_tmp_inline348__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v631 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %post_tmp_inline348__ssa_v0
      ;
      uint64_t v632 = (uint64_t) v47;
      TASSIGN(v631, v632);
      // pto: %t__tmp_v150
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v633 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v84);
      // pto: %t__tmp_v150
      ;
      uint64_t v634 = (uint64_t) v58;
      TASSIGN(v633, v634);
      // pto: %post_values_inline347__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v635 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %post_values_inline347__ssa_v0
      ;
      uint64_t v636 = (uint64_t) v48;
      TASSIGN(v635, v636);
      pipe_barrier(PIPE_V);
      TGATHER(v635, v633, v627, v631);
      // pto: %t__tmp_v151
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v637 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %t__tmp_v151
      ;
      uint64_t v638 = (uint64_t) v36;
      TASSIGN(v637, v638);
      pipe_barrier(PIPE_V);
      TMULS(v637, v635, v93);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      // pto: %138
      ;
      int64_t v639 = (int64_t) ((uint64_t) v485 * (uint64_t) v94);
      v637.SetValidShape(v66, v639);
      // pto: %139
      ;
      bool v640 = v457 == v76;
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      if (v640) {
        // pto: %141
        ;
        int64_t v641 = (int64_t) ((uint64_t) v483 * (uint64_t) v94);
        // pto: %142
        ;
        int64_t v642 = v641 < v76 ? v76 : v641;
        // pto: %post_flat_inline541__ssa_v0_pview
        ;
        const int64_t v643 = 0;
        // pto: %post_flat_inline541__ssa_v0_pview
        ;
        __gm__ float* v644 = PTOAS__GLOBAL_TENSOR_DATA(v166);
        // pto: %post_flat_inline541__ssa_v0_pview
        ;
        const int64_t v645 = 1;
        // pto: %post_flat_inline541__ssa_v0_pview
        ;
        const int64_t v646 = 1;
        // pto: %post_flat_inline541__ssa_v0_pview
        ;
        const int64_t v647 = 1;
        // pto: %post_flat_inline541__ssa_v0_pview
        ;
        int64_t v648 = v66 * v158;
        // pto: %post_flat_inline541__ssa_v0_pview
        ;
        int64_t v649 = v647 * v648;
        // pto: %post_flat_inline541__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, -1> v650 = pto::Shape<1, 1, 1, 1, -1>(v645, v646, v647, v66, v639);
        // pto: %post_flat_inline541__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v651 = pto::Stride<-1, -1, -1, -1, -1>(v646 * v649, v649, v648, v158, v66);
        // pto: %post_flat_inline541__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v652 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v644 + (v643 + v76 * v158 + v642 * v66), v650, v651);
        TSTORE(v652, v637);
      };
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      // pto: %143
      ;
      uint64_t v653 = (uint64_t) v457;
      // pto: %143
      ;
      int64_t v654 = (int64_t) (v653 * (uint64_t) v78);
      // pto: %144
      ;
      uint64_t v655 = (uint64_t) v458;
      // pto: %144
      ;
      uint64_t v656 = (uint64_t) v457;
      // pto: %144, %145
      ;
      int64_t v657 = (int64_t) ((uint64_t) ((int64_t) v655 - v656) * (uint64_t) v78);
      // pto: %146
      ;
      int64_t v658 = v485 / v77;
      // pto: %147
      ;
      int64_t v659 = (int64_t) ((uint64_t) v658 * (uint64_t) v77);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      for (int64_t v660 = v76; v660 < v659; v660 += v77) {
        // pto: %148
        ;
        int64_t v661 = (int64_t) ((uint64_t) v483 + (uint64_t) v660);
        // pto: %149
        ;
        int64_t v662 = (int64_t) ((uint64_t) v660 + (uint64_t) v66);
        // pto: %150
        ;
        int64_t v663 = (int64_t) ((uint64_t) v483 + (uint64_t) v662);
        // pto: %flat_offset_mul
        ;
        int64_t v664 = (int64_t) ((uint64_t) v660 * (uint64_t) v80);
        // pto: %t__tmp_v152
        ;
        float v665 = v599.GetValue(v664);
        // pto: %151
        ;
        float v666 = v665 + v82;
        // pto: %153, %t__tmp_v153
        ;
        float v667 = v599.GetValue((int64_t) ((uint64_t) v664 + (uint64_t) v66));
        // pto: %154
        ;
        float v668 = v667 + v82;
        // pto: %156, %t__tmp_v154
        ;
        float v669 = v599.GetValue((int64_t) ((uint64_t) v664 + (uint64_t) v77));
        // pto: %157
        ;
        float v670 = v669 + v82;
        // pto: %159, %t__tmp_v155
        ;
        float v671 = v599.GetValue((int64_t) ((uint64_t) v664 + (uint64_t) v69));
        // pto: %160
        ;
        float v672 = v671 + v82;
        // pto: %t__tmp_v156
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v673 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %t__tmp_v156
        ;
        uint64_t v674 = (uint64_t) v38;
        TASSIGN(v673, v674);
        // pto: %161
        ;
        int64_t v675 = v661 < v76 ? v76 : v661;
        // pto: %162
        ;
        int64_t v676 = v654 < v76 ? v76 : v654;
        // pto: %163
        ;
        const int64_t v677 = 0;
        // pto: %163
        ;
        __gm__ bfloat16_t* v678 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %163
        ;
        const int64_t v679 = 1;
        // pto: %163
        ;
        const int64_t v680 = 1;
        // pto: %163
        ;
        const int64_t v681 = 1;
        // pto: %163
        ;
        int64_t v682 = v66 * v68;
        // pto: %163
        ;
        int64_t v683 = v681 * v682;
        // pto: %163
        ;
        pto::Shape<1, 1, 1, 1, -1> v684 = pto::Shape<1, 1, 1, 1, -1>(v679, v680, v681, v66, v657);
        // pto: %163
        ;
        pto::Stride<-1, -1, -1, -1, -1> v685 = pto::Stride<-1, -1, -1, -1, -1>(v680 * v683, v683, v682, v68, v66);
        // pto: %163
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v686 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v678 + (v677 + v675 * v68 + v676 * v66), v684, v685);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v673, v686);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        // pto: %t__tmp_v157
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v687 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %t__tmp_v157
        ;
        uint64_t v688 = (uint64_t) v59;
        TASSIGN(v687, v688);
        // pto: %165
        ;
        int64_t v689 = (int64_t) ((uint64_t) v654 + (uint64_t) v71);
        // pto: %166
        ;
        int64_t v690 = v689 < v76 ? v76 : v689;
        // pto: %167
        ;
        const int64_t v691 = 0;
        // pto: %167
        ;
        __gm__ bfloat16_t* v692 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %167
        ;
        const int64_t v693 = 1;
        // pto: %167
        ;
        const int64_t v694 = 1;
        // pto: %167
        ;
        const int64_t v695 = 1;
        // pto: %167
        ;
        int64_t v696 = v66 * v68;
        // pto: %167
        ;
        int64_t v697 = v695 * v696;
        // pto: %167
        ;
        pto::Shape<1, 1, 1, 1, -1> v698 = pto::Shape<1, 1, 1, 1, -1>(v693, v694, v695, v66, v657);
        // pto: %167
        ;
        pto::Stride<-1, -1, -1, -1, -1> v699 = pto::Stride<-1, -1, -1, -1, -1>(v694 * v697, v697, v696, v68, v66);
        // pto: %167
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v700 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v692 + (v691 + v675 * v68 + v690 * v66), v698, v699);
        TLOAD(v687, v700);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %t__tmp_v158
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v701 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %t__tmp_v158
        ;
        uint64_t v702 = (uint64_t) v60;
        TASSIGN(v701, v702);
        // pto: %169
        ;
        int64_t v703 = (int64_t) ((uint64_t) v654 + (uint64_t) v95);
        // pto: %170
        ;
        int64_t v704 = v703 < v76 ? v76 : v703;
        // pto: %171
        ;
        const int64_t v705 = 0;
        // pto: %171
        ;
        __gm__ bfloat16_t* v706 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %171
        ;
        const int64_t v707 = 1;
        // pto: %171
        ;
        const int64_t v708 = 1;
        // pto: %171
        ;
        const int64_t v709 = 1;
        // pto: %171
        ;
        int64_t v710 = v66 * v68;
        // pto: %171
        ;
        int64_t v711 = v709 * v710;
        // pto: %171
        ;
        pto::Shape<1, 1, 1, 1, -1> v712 = pto::Shape<1, 1, 1, 1, -1>(v707, v708, v709, v66, v657);
        // pto: %171
        ;
        pto::Stride<-1, -1, -1, -1, -1> v713 = pto::Stride<-1, -1, -1, -1, -1>(v708 * v711, v711, v710, v68, v66);
        // pto: %171
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v714 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v706 + (v705 + v675 * v68 + v704 * v66), v712, v713);
        TLOAD(v701, v714);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %t__tmp_v159
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v715 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %t__tmp_v159
        ;
        uint64_t v716 = (uint64_t) v34;
        TASSIGN(v715, v716);
        // pto: %173
        ;
        int64_t v717 = (int64_t) ((uint64_t) v654 + (uint64_t) v96);
        // pto: %174
        ;
        int64_t v718 = v717 < v76 ? v76 : v717;
        // pto: %175
        ;
        const int64_t v719 = 0;
        // pto: %175
        ;
        __gm__ bfloat16_t* v720 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %175
        ;
        const int64_t v721 = 1;
        // pto: %175
        ;
        const int64_t v722 = 1;
        // pto: %175
        ;
        const int64_t v723 = 1;
        // pto: %175
        ;
        int64_t v724 = v66 * v68;
        // pto: %175
        ;
        int64_t v725 = v723 * v724;
        // pto: %175
        ;
        pto::Shape<1, 1, 1, 1, -1> v726 = pto::Shape<1, 1, 1, 1, -1>(v721, v722, v723, v66, v657);
        // pto: %175
        ;
        pto::Stride<-1, -1, -1, -1, -1> v727 = pto::Stride<-1, -1, -1, -1, -1>(v722 * v725, v725, v724, v68, v66);
        // pto: %175
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v728 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v720 + (v719 + v675 * v68 + v718 * v66), v726, v727);
        TLOAD(v715, v728);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %178
        ;
        int64_t v729 = (int64_t) ((uint64_t) v662 * (uint64_t) v80);
        // pto: %176
        ;
        float v730 = v599.GetValue(v729);
        // pto: %180
        ;
        float v731 = v730 + v82;
        // pto: %184, %181
        ;
        float v732 = v599.GetValue((int64_t) ((uint64_t) v729 + (uint64_t) v66));
        // pto: %185
        ;
        float v733 = v732 + v82;
        // pto: %189, %186
        ;
        float v734 = v599.GetValue((int64_t) ((uint64_t) v729 + (uint64_t) v77));
        // pto: %190
        ;
        float v735 = v734 + v82;
        // pto: %194, %191
        ;
        float v736 = v599.GetValue((int64_t) ((uint64_t) v729 + (uint64_t) v69));
        // pto: %195
        ;
        float v737 = v736 + v82;
        // pto: %4
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v738 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %4
        ;
        uint64_t v739 = (uint64_t) v61;
        TASSIGN(v738, v739);
        // pto: %196
        ;
        int64_t v740 = v663 < v76 ? v76 : v663;
        // pto: %198
        ;
        const int64_t v741 = 0;
        // pto: %198
        ;
        __gm__ bfloat16_t* v742 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %198
        ;
        const int64_t v743 = 1;
        // pto: %198
        ;
        const int64_t v744 = 1;
        // pto: %198
        ;
        const int64_t v745 = 1;
        // pto: %198
        ;
        int64_t v746 = v66 * v68;
        // pto: %198
        ;
        int64_t v747 = v745 * v746;
        // pto: %198
        ;
        pto::Shape<1, 1, 1, 1, -1> v748 = pto::Shape<1, 1, 1, 1, -1>(v743, v744, v745, v66, v657);
        // pto: %198
        ;
        pto::Stride<-1, -1, -1, -1, -1> v749 = pto::Stride<-1, -1, -1, -1, -1>(v744 * v747, v747, v746, v68, v66);
        // pto: %198
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v750 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v742 + (v741 + v740 * v68 + v676 * v66), v748, v749);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        TLOAD(v738, v750);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %5
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v751 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %5
        ;
        uint64_t v752 = (uint64_t) v62;
        TASSIGN(v751, v752);
        // pto: %202
        ;
        const int64_t v753 = 0;
        // pto: %202
        ;
        __gm__ bfloat16_t* v754 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %202
        ;
        const int64_t v755 = 1;
        // pto: %202
        ;
        const int64_t v756 = 1;
        // pto: %202
        ;
        const int64_t v757 = 1;
        // pto: %202
        ;
        int64_t v758 = v66 * v68;
        // pto: %202
        ;
        int64_t v759 = v757 * v758;
        // pto: %202
        ;
        pto::Shape<1, 1, 1, 1, -1> v760 = pto::Shape<1, 1, 1, 1, -1>(v755, v756, v757, v66, v657);
        // pto: %202
        ;
        pto::Stride<-1, -1, -1, -1, -1> v761 = pto::Stride<-1, -1, -1, -1, -1>(v756 * v759, v759, v758, v68, v66);
        // pto: %202
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v762 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v754 + (v753 + v740 * v68 + v690 * v66), v760, v761);
        TLOAD(v751, v762);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        // pto: %6
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v763 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %6
        ;
        uint64_t v764 = (uint64_t) v63;
        TASSIGN(v763, v764);
        // pto: %206
        ;
        const int64_t v765 = 0;
        // pto: %206
        ;
        __gm__ bfloat16_t* v766 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %206
        ;
        const int64_t v767 = 1;
        // pto: %206
        ;
        const int64_t v768 = 1;
        // pto: %206
        ;
        const int64_t v769 = 1;
        // pto: %206
        ;
        int64_t v770 = v66 * v68;
        // pto: %206
        ;
        int64_t v771 = v769 * v770;
        // pto: %206
        ;
        pto::Shape<1, 1, 1, 1, -1> v772 = pto::Shape<1, 1, 1, 1, -1>(v767, v768, v769, v66, v657);
        // pto: %206
        ;
        pto::Stride<-1, -1, -1, -1, -1> v773 = pto::Stride<-1, -1, -1, -1, -1>(v768 * v771, v771, v770, v68, v66);
        // pto: %206
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v774 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v766 + (v765 + v740 * v68 + v704 * v66), v772, v773);
        TLOAD(v763, v774);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        // pto: %7
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v775 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %7
        ;
        uint64_t v776 = (uint64_t) v64;
        TASSIGN(v775, v776);
        // pto: %210
        ;
        const int64_t v777 = 0;
        // pto: %210
        ;
        __gm__ bfloat16_t* v778 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %210
        ;
        const int64_t v779 = 1;
        // pto: %210
        ;
        const int64_t v780 = 1;
        // pto: %210
        ;
        const int64_t v781 = 1;
        // pto: %210
        ;
        int64_t v782 = v66 * v68;
        // pto: %210
        ;
        int64_t v783 = v781 * v782;
        // pto: %210
        ;
        pto::Shape<1, 1, 1, 1, -1> v784 = pto::Shape<1, 1, 1, 1, -1>(v779, v780, v781, v66, v657);
        // pto: %210
        ;
        pto::Stride<-1, -1, -1, -1, -1> v785 = pto::Stride<-1, -1, -1, -1, -1>(v780 * v783, v783, v782, v68, v66);
        // pto: %210
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v786 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v778 + (v777 + v740 * v68 + v718 * v66), v784, v785);
        TLOAD(v775, v786);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        // pto: %part_scalar_x0_inline506__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v787 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %part_scalar_x0_inline506__ssa_v0
        ;
        uint64_t v788 = (uint64_t) v36;
        TASSIGN(v787, v788);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        RoundMode v789 = RoundMode::CAST_ROUND;
        SaturationMode v790 = SaturationMode::ON;
        TCVT(v787, v673, v789, v790);
        // pto: %part_scalar_y0_inline342__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v791 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %part_scalar_y0_inline342__ssa_v0
        ;
        uint64_t v792 = (uint64_t) v36;
        TASSIGN(v791, v792);
        pipe_barrier(PIPE_V);
        TMULS(v791, v787, v666);
        // pto: %part_scalar_x1_inline578__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v793 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %part_scalar_x1_inline578__ssa_v0
        ;
        uint64_t v794 = (uint64_t) v38;
        TASSIGN(v793, v794);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v795 = RoundMode::CAST_ROUND;
        SaturationMode v796 = SaturationMode::ON;
        TCVT(v793, v687, v795, v796);
        // pto: %part_scalar_y1_inline458__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v797 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %part_scalar_y1_inline458__ssa_v0
        ;
        uint64_t v798 = (uint64_t) v38;
        TASSIGN(v797, v798);
        pipe_barrier(PIPE_V);
        TMULS(v797, v793, v668);
        // pto: %part_scalar_sum01_inline341__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v799 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %part_scalar_sum01_inline341__ssa_v0
        ;
        uint64_t v800 = (uint64_t) v36;
        TASSIGN(v799, v800);
        pipe_barrier(PIPE_V);
        TADD(v799, v791, v797);
        // pto: %part_scalar_x2_inline340__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v801 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %part_scalar_x2_inline340__ssa_v0
        ;
        uint64_t v802 = (uint64_t) v38;
        TASSIGN(v801, v802);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        RoundMode v803 = RoundMode::CAST_ROUND;
        SaturationMode v804 = SaturationMode::ON;
        TCVT(v801, v701, v803, v804);
        // pto: %part_scalar_y2_inline339__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v805 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %part_scalar_y2_inline339__ssa_v0
        ;
        uint64_t v806 = (uint64_t) v38;
        TASSIGN(v805, v806);
        pipe_barrier(PIPE_V);
        TMULS(v805, v801, v670);
        // pto: %part_scalar_x3_inline528__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v807 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %part_scalar_x3_inline528__ssa_v0
        ;
        uint64_t v808 = (uint64_t) v59;
        TASSIGN(v807, v808);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        RoundMode v809 = RoundMode::CAST_ROUND;
        SaturationMode v810 = SaturationMode::ON;
        TCVT(v807, v715, v809, v810);
        // pto: %part_scalar_y3_inline338__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v811 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %part_scalar_y3_inline338__ssa_v0
        ;
        uint64_t v812 = (uint64_t) v59;
        TASSIGN(v811, v812);
        pipe_barrier(PIPE_V);
        TMULS(v811, v807, v672);
        // pto: %t__tmp_v160
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v813 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %t__tmp_v160
        ;
        uint64_t v814 = (uint64_t) v38;
        TASSIGN(v813, v814);
        pipe_barrier(PIPE_V);
        TADD(v813, v805, v811);
        // pto: %part_scalar_result_inline337__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v815 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %part_scalar_result_inline337__ssa_v0
        ;
        uint64_t v816 = (uint64_t) v36;
        TASSIGN(v815, v816);
        pipe_barrier(PIPE_V);
        TADD(v815, v799, v813);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %part_scalar_bf16_inline558__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v817 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %part_scalar_bf16_inline558__ssa_v0
        ;
        uint64_t v818 = (uint64_t) v36;
        TASSIGN(v817, v818);
        pipe_barrier(PIPE_V);
        RoundMode v819 = RoundMode::CAST_RINT;
        SaturationMode v820 = SaturationMode::ON;
        TCVT(v817, v815, v819, v820);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        v817.SetValidShape(v66, v657);
        // pto: %mixed_view_inline442__ssa_v0_pview
        ;
        const int64_t v821 = 0;
        // pto: %mixed_view_inline442__ssa_v0_pview
        ;
        __gm__ bfloat16_t* v822 = PTOAS__GLOBAL_TENSOR_DATA(v175);
        // pto: %mixed_view_inline442__ssa_v0_pview
        ;
        const int64_t v823 = 1;
        // pto: %mixed_view_inline442__ssa_v0_pview
        ;
        const int64_t v824 = 1;
        // pto: %mixed_view_inline442__ssa_v0_pview
        ;
        const int64_t v825 = 1;
        // pto: %mixed_view_inline442__ssa_v0_pview
        ;
        int64_t v826 = v66 * v71;
        // pto: %mixed_view_inline442__ssa_v0_pview
        ;
        int64_t v827 = v825 * v826;
        // pto: %mixed_view_inline442__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, -1> v828 = pto::Shape<1, 1, 1, 1, -1>(v823, v824, v825, v66, v657);
        // pto: %mixed_view_inline442__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v829 = pto::Stride<-1, -1, -1, -1, -1>(v824 * v827, v827, v826, v71, v66);
        // pto: %mixed_view_inline442__ssa_v0_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v830 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v822 + (v821 + v675 * v71 + v676 * v66), v828, v829);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v830, v817);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        // pto: %8
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v831 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %8
        ;
        uint64_t v832 = (uint64_t) v35;
        TASSIGN(v831, v832);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        RoundMode v833 = RoundMode::CAST_ROUND;
        SaturationMode v834 = SaturationMode::ON;
        TCVT(v831, v738, v833, v834);
        // pto: %9
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v835 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %9
        ;
        uint64_t v836 = (uint64_t) v35;
        TASSIGN(v835, v836);
        pipe_barrier(PIPE_V);
        TMULS(v835, v831, v731);
        // pto: %10
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v837 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %10
        ;
        uint64_t v838 = (uint64_t) v61;
        TASSIGN(v837, v838);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        RoundMode v839 = RoundMode::CAST_ROUND;
        SaturationMode v840 = SaturationMode::ON;
        TCVT(v837, v751, v839, v840);
        // pto: %11
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v841 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %11
        ;
        uint64_t v842 = (uint64_t) v61;
        TASSIGN(v841, v842);
        pipe_barrier(PIPE_V);
        TMULS(v841, v837, v733);
        // pto: %12
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v843 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %12
        ;
        uint64_t v844 = (uint64_t) v35;
        TASSIGN(v843, v844);
        pipe_barrier(PIPE_V);
        TADD(v843, v835, v841);
        // pto: %13
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v845 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %13
        ;
        uint64_t v846 = (uint64_t) v61;
        TASSIGN(v845, v846);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        RoundMode v847 = RoundMode::CAST_ROUND;
        SaturationMode v848 = SaturationMode::ON;
        TCVT(v845, v763, v847, v848);
        // pto: %14
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v849 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %14
        ;
        uint64_t v850 = (uint64_t) v61;
        TASSIGN(v849, v850);
        pipe_barrier(PIPE_V);
        TMULS(v849, v845, v735);
        // pto: %15
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v851 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %15
        ;
        uint64_t v852 = (uint64_t) v62;
        TASSIGN(v851, v852);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        RoundMode v853 = RoundMode::CAST_ROUND;
        SaturationMode v854 = SaturationMode::ON;
        TCVT(v851, v775, v853, v854);
        // pto: %16
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v855 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %16
        ;
        uint64_t v856 = (uint64_t) v62;
        TASSIGN(v855, v856);
        pipe_barrier(PIPE_V);
        TMULS(v855, v851, v737);
        // pto: %17
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v857 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %17
        ;
        uint64_t v858 = (uint64_t) v61;
        TASSIGN(v857, v858);
        pipe_barrier(PIPE_V);
        TADD(v857, v849, v855);
        // pto: %18
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v859 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %18
        ;
        uint64_t v860 = (uint64_t) v35;
        TASSIGN(v859, v860);
        pipe_barrier(PIPE_V);
        TADD(v859, v843, v857);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        // pto: %19
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v861 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %19
        ;
        uint64_t v862 = (uint64_t) v35;
        TASSIGN(v861, v862);
        pipe_barrier(PIPE_V);
        RoundMode v863 = RoundMode::CAST_RINT;
        SaturationMode v864 = SaturationMode::ON;
        TCVT(v861, v859, v863, v864);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        v861.SetValidShape(v66, v657);
        // pto: %216
        ;
        const int64_t v865 = 0;
        // pto: %216
        ;
        __gm__ bfloat16_t* v866 = PTOAS__GLOBAL_TENSOR_DATA(v175);
        // pto: %216
        ;
        const int64_t v867 = 1;
        // pto: %216
        ;
        const int64_t v868 = 1;
        // pto: %216
        ;
        const int64_t v869 = 1;
        // pto: %216
        ;
        int64_t v870 = v66 * v71;
        // pto: %216
        ;
        int64_t v871 = v869 * v870;
        // pto: %216
        ;
        pto::Shape<1, 1, 1, 1, -1> v872 = pto::Shape<1, 1, 1, 1, -1>(v867, v868, v869, v66, v657);
        // pto: %216
        ;
        pto::Stride<-1, -1, -1, -1, -1> v873 = pto::Stride<-1, -1, -1, -1, -1>(v868 * v871, v871, v870, v71, v66);
        // pto: %216
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v874 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v866 + (v865 + v740 * v71 + v676 * v66), v872, v873);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v874, v861);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
      };
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_S, PIPE_V, EVENT_ID1);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
      // pto: %217, %218
      ;
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
      if ((int64_t) ((uint64_t) v485 - (uint64_t) v659) == v66) {
        // pto: %219
        ;
        int64_t v875 = (int64_t) ((uint64_t) v483 + (uint64_t) v659);
        int64_t v876 = (int64_t) ((uint64_t) v658 * (uint64_t) v70);
        // pto: %220
        ;
        float v877 = v599.GetValue(v876);
        // pto: %223
        ;
        float v878 = v877 + v82;
        // pto: %226, %224
        ;
        float v879 = v599.GetValue((int64_t) ((uint64_t) v876 + (uint64_t) v66));
        // pto: %227
        ;
        float v880 = v879 + v82;
        // pto: %230, %228
        ;
        float v881 = v599.GetValue((int64_t) ((uint64_t) v876 + (uint64_t) v77));
        // pto: %231
        ;
        float v882 = v881 + v82;
        // pto: %234, %232
        ;
        float v883 = v599.GetValue((int64_t) ((uint64_t) v876 + (uint64_t) v69));
        // pto: %235
        ;
        float v884 = v883 + v82;
        // pto: %21
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v885 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %21
        ;
        uint64_t v886 = (uint64_t) v38;
        TASSIGN(v885, v886);
        // pto: %236
        ;
        int64_t v887 = v875 < v76 ? v76 : v875;
        // pto: %237
        ;
        int64_t v888 = v654 < v76 ? v76 : v654;
        // pto: %238
        ;
        const int64_t v889 = 0;
        // pto: %238
        ;
        __gm__ bfloat16_t* v890 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %238
        ;
        const int64_t v891 = 1;
        // pto: %238
        ;
        const int64_t v892 = 1;
        // pto: %238
        ;
        const int64_t v893 = 1;
        // pto: %238
        ;
        int64_t v894 = v66 * v68;
        // pto: %238
        ;
        int64_t v895 = v893 * v894;
        // pto: %238
        ;
        pto::Shape<1, 1, 1, 1, -1> v896 = pto::Shape<1, 1, 1, 1, -1>(v891, v892, v893, v66, v657);
        // pto: %238
        ;
        pto::Stride<-1, -1, -1, -1, -1> v897 = pto::Stride<-1, -1, -1, -1, -1>(v892 * v895, v895, v894, v68, v66);
        // pto: %238
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v898 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v890 + (v889 + v887 * v68 + v888 * v66), v896, v897);
        TLOAD(v885, v898);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %22
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v899 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %22
        ;
        uint64_t v900 = (uint64_t) v36;
        TASSIGN(v899, v900);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v901 = RoundMode::CAST_ROUND;
        SaturationMode v902 = SaturationMode::ON;
        TCVT(v899, v885, v901, v902);
        // pto: %23
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v903 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %23
        ;
        uint64_t v904 = (uint64_t) v36;
        TASSIGN(v903, v904);
        pipe_barrier(PIPE_V);
        TMULS(v903, v899, v878);
        // pto: %24
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v905 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %24
        ;
        uint64_t v906 = (uint64_t) v59;
        TASSIGN(v905, v906);
        // pto: %240
        ;
        int64_t v907 = (int64_t) ((uint64_t) v654 + (uint64_t) v71);
        // pto: %241
        ;
        int64_t v908 = v907 < v76 ? v76 : v907;
        // pto: %242
        ;
        const int64_t v909 = 0;
        // pto: %242
        ;
        __gm__ bfloat16_t* v910 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %242
        ;
        const int64_t v911 = 1;
        // pto: %242
        ;
        const int64_t v912 = 1;
        // pto: %242
        ;
        const int64_t v913 = 1;
        // pto: %242
        ;
        int64_t v914 = v66 * v68;
        // pto: %242
        ;
        int64_t v915 = v913 * v914;
        // pto: %242
        ;
        pto::Shape<1, 1, 1, 1, -1> v916 = pto::Shape<1, 1, 1, 1, -1>(v911, v912, v913, v66, v657);
        // pto: %242
        ;
        pto::Stride<-1, -1, -1, -1, -1> v917 = pto::Stride<-1, -1, -1, -1, -1>(v912 * v915, v915, v914, v68, v66);
        // pto: %242
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v918 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v910 + (v909 + v887 * v68 + v908 * v66), v916, v917);
        TLOAD(v905, v918);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %25
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v919 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %25
        ;
        uint64_t v920 = (uint64_t) v38;
        TASSIGN(v919, v920);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v921 = RoundMode::CAST_ROUND;
        SaturationMode v922 = SaturationMode::ON;
        TCVT(v919, v905, v921, v922);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %26
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v923 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %26
        ;
        uint64_t v924 = (uint64_t) v38;
        TASSIGN(v923, v924);
        pipe_barrier(PIPE_V);
        TMULS(v923, v919, v880);
        // pto: %27
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v925 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %27
        ;
        uint64_t v926 = (uint64_t) v36;
        TASSIGN(v925, v926);
        pipe_barrier(PIPE_V);
        TADD(v925, v903, v923);
        // pto: %28
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v927 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %28
        ;
        uint64_t v928 = (uint64_t) v59;
        TASSIGN(v927, v928);
        // pto: %244
        ;
        int64_t v929 = (int64_t) ((uint64_t) v654 + (uint64_t) v95);
        // pto: %245
        ;
        int64_t v930 = v929 < v76 ? v76 : v929;
        // pto: %246
        ;
        const int64_t v931 = 0;
        // pto: %246
        ;
        __gm__ bfloat16_t* v932 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %246
        ;
        const int64_t v933 = 1;
        // pto: %246
        ;
        const int64_t v934 = 1;
        // pto: %246
        ;
        const int64_t v935 = 1;
        // pto: %246
        ;
        int64_t v936 = v66 * v68;
        // pto: %246
        ;
        int64_t v937 = v935 * v936;
        // pto: %246
        ;
        pto::Shape<1, 1, 1, 1, -1> v938 = pto::Shape<1, 1, 1, 1, -1>(v933, v934, v935, v66, v657);
        // pto: %246
        ;
        pto::Stride<-1, -1, -1, -1, -1> v939 = pto::Stride<-1, -1, -1, -1, -1>(v934 * v937, v937, v936, v68, v66);
        // pto: %246
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v940 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v932 + (v931 + v887 * v68 + v930 * v66), v938, v939);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v927, v940);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %29
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v941 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %29
        ;
        uint64_t v942 = (uint64_t) v38;
        TASSIGN(v941, v942);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v943 = RoundMode::CAST_ROUND;
        SaturationMode v944 = SaturationMode::ON;
        TCVT(v941, v927, v943, v944);
        // pto: %30
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v945 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %30
        ;
        uint64_t v946 = (uint64_t) v38;
        TASSIGN(v945, v946);
        pipe_barrier(PIPE_V);
        TMULS(v945, v941, v882);
        // pto: %31
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v947 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %31
        ;
        uint64_t v948 = (uint64_t) v35;
        TASSIGN(v947, v948);
        // pto: %248
        ;
        int64_t v949 = (int64_t) ((uint64_t) v654 + (uint64_t) v96);
        // pto: %249
        ;
        int64_t v950 = v949 < v76 ? v76 : v949;
        // pto: %250
        ;
        const int64_t v951 = 0;
        // pto: %250
        ;
        __gm__ bfloat16_t* v952 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %250
        ;
        const int64_t v953 = 1;
        // pto: %250
        ;
        const int64_t v954 = 1;
        // pto: %250
        ;
        const int64_t v955 = 1;
        // pto: %250
        ;
        int64_t v956 = v66 * v68;
        // pto: %250
        ;
        int64_t v957 = v955 * v956;
        // pto: %250
        ;
        pto::Shape<1, 1, 1, 1, -1> v958 = pto::Shape<1, 1, 1, 1, -1>(v953, v954, v955, v66, v657);
        // pto: %250
        ;
        pto::Stride<-1, -1, -1, -1, -1> v959 = pto::Stride<-1, -1, -1, -1, -1>(v954 * v957, v957, v956, v68, v66);
        // pto: %250
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v960 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v952 + (v951 + v887 * v68 + v950 * v66), v958, v959);
        TLOAD(v947, v960);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %32
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v961 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %32
        ;
        uint64_t v962 = (uint64_t) v59;
        TASSIGN(v961, v962);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v963 = RoundMode::CAST_ROUND;
        SaturationMode v964 = SaturationMode::ON;
        TCVT(v961, v947, v963, v964);
        // pto: %33
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v965 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %33
        ;
        uint64_t v966 = (uint64_t) v59;
        TASSIGN(v965, v966);
        pipe_barrier(PIPE_V);
        TMULS(v965, v961, v884);
        // pto: %34
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v967 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %34
        ;
        uint64_t v968 = (uint64_t) v38;
        TASSIGN(v967, v968);
        pipe_barrier(PIPE_V);
        TADD(v967, v945, v965);
        // pto: %35
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v969 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %35
        ;
        uint64_t v970 = (uint64_t) v36;
        TASSIGN(v969, v970);
        pipe_barrier(PIPE_V);
        TADD(v969, v925, v967);
        // pto: %36
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v971 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v657);
        // pto: %36
        ;
        uint64_t v972 = (uint64_t) v36;
        TASSIGN(v971, v972);
        pipe_barrier(PIPE_V);
        RoundMode v973 = RoundMode::CAST_RINT;
        SaturationMode v974 = SaturationMode::ON;
        TCVT(v971, v969, v973, v974);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        v971.SetValidShape(v66, v657);
        // pto: %254
        ;
        const int64_t v975 = 0;
        // pto: %254
        ;
        __gm__ bfloat16_t* v976 = PTOAS__GLOBAL_TENSOR_DATA(v175);
        // pto: %254
        ;
        const int64_t v977 = 1;
        // pto: %254
        ;
        const int64_t v978 = 1;
        // pto: %254
        ;
        const int64_t v979 = 1;
        // pto: %254
        ;
        int64_t v980 = v66 * v71;
        // pto: %254
        ;
        int64_t v981 = v979 * v980;
        // pto: %254
        ;
        pto::Shape<1, 1, 1, 1, -1> v982 = pto::Shape<1, 1, 1, 1, -1>(v977, v978, v979, v66, v657);
        // pto: %254
        ;
        pto::Stride<-1, -1, -1, -1, -1> v983 = pto::Stride<-1, -1, -1, -1, -1>(v978 * v981, v981, v980, v71, v66);
        // pto: %254
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v984 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v976 + (v975 + v887 * v71 + v888 * v66), v982, v983);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v984, v971);
      };
      set_flag(PIPE_S, PIPE_V, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    };
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  }
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}