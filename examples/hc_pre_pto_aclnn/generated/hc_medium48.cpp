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
  // pto: %mixes_partials_inline426__ssa_v0_view
  const int64_t v53 = 1;
  // pto: %mixes_partials_inline426__ssa_v0_view
  const int64_t v54 = 1;
  // pto: %mixes_partials_inline426__ssa_v0_view
  const int64_t v55 = 1;
  // pto: %mixes_partials_inline426__ssa_v0_view
  int64_t v56 = v52 * v39;
  // pto: %mixes_partials_inline426__ssa_v0_view
  int64_t v57 = v55 * v56;
  // pto: %mixes_partials_inline426__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v58 = pto::Shape<1, 1, 1, -1, -1>(v53, v54, v55, v52, v39);
  // pto: %mixes_partials_inline426__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v59 = pto::Stride<-1, -1, -1, -1, -1>(v54 * v57, v57, v56, v39, v40);
  // pto: %mixes_partials_inline426__ssa_v0_view
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
  // pto: %worker_inline466__ssa_v0
  // pto: %linear_workers_inline381__ssa_v0_idx
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
      // pto: %mid_weight_chunk_inline622__tile
      ;
      Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v80 = Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v41, v48);
      // pto: %mid_weight_chunk_inline622__tile
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
      // pto: %mid_gathered_chunk_inline372__tile
      ;
      const int32_t v93 = 48;
      // pto: %mid_gathered_chunk_inline372__tile
      ;
      const int32_t v94 = 512;
      // pto: %mid_gathered_chunk_inline372__tile
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
        // pto: %mid_weight_chunk_inline622__tile_t
        ;
        Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v48, v41);
        // pto: %mid_weight_chunk_inline622__tile_t
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
    // pto: %mixes_partials_inline426__iter_v1_pview
    ;
    __gm__ float* v149 = PTOAS__GLOBAL_TENSOR_DATA(v60);
    // pto: %mixes_partials_inline426__iter_v1_pview
    ;
    const int64_t v150 = 0;
    // pto: %mixes_partials_inline426__iter_v1_pview
    ;
    const int64_t v151 = 32;
    // pto: %mixes_partials_inline426__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 48, 32> v152 = pto::Shape<1, 1, 1, 48, 32>();
    // pto: %mixes_partials_inline426__iter_v1_pview
    ;
    pto::Stride<1536, 1536, 1536, 32, 1> v153 = pto::Stride<1536, 1536, 1536, 32, 1>();
    // pto: %mixes_partials_inline426__iter_v1_pview
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
  // pto: %c17664_i64
  const int64_t v38 = 17664;
  // pto: %c18176_i64
  const int64_t v39 = 18176;
  // pto: %c17728_i64
  const int64_t v40 = 17728;
  const int64_t v41 = 16672;
  // pto: %c17152_i64
  const int64_t v42 = 17152;
  // pto: %c17408_i64
  const int64_t v43 = 17408;
  // pto: %c17920_i64
  const int64_t v44 = 17920;
  // pto: %c16768_i64
  const int64_t v45 = 16768;
  // pto: %c16896_i64
  const int64_t v46 = 16896;
  // pto: %c17024_i64
  const int64_t v47 = 17024;
  // pto: %c17280_i64
  const int64_t v48 = 17280;
  // pto: %c18048_i64
  const int64_t v49 = 18048;
  // pto: %c17536_i64
  const int64_t v50 = 17536;
  // pto: %c18432_i64
  const int64_t v51 = 18432;
  // pto: %c18464_i64
  const int64_t v52 = 18464;
  // pto: %c18496_i64
  const int64_t v53 = 18496;
  // pto: %c18528_i64
  const int64_t v54 = 18528;
  // pto: %c17696_i64
  const int64_t v55 = 17696;
  // pto: %c0_i64
  const int64_t v56 = 0;
  // pto: %c33024_i64
  const int64_t v57 = 33024;
  // pto: %c49408_i64
  const int64_t v58 = 49408;
  // pto: %c57600_i64
  const int64_t v59 = 57600;
  // pto: %c82176_i64
  const int64_t v60 = 82176;
  // pto: %c98560_i64
  const int64_t v61 = 98560;
  // pto: %c106752_i64
  const int64_t v62 = 106752;
  // pto: %c8448_i64
  const int64_t v63 = 8448;
  // pto: %c32_index
  const int64_t v64 = 32;
  // pto: %c1_index
  const int64_t v65 = 1;
  // pto: %c24_index
  const int64_t v66 = 24;
  // pto: %c16384_index
  const int64_t v67 = 16384;
  // pto: %c3_index
  const int64_t v68 = 3;
  const int64_t v69 = 16;
  // pto: %c4096_index
  const int64_t v70 = 4096;
  // pto: %c0_i32
  const int32_t v71 = 0;
  // pto: %c48_index
  const int64_t v72 = 48;
  // pto: %c1024_index
  const int64_t v73 = 1024;
  // pto: %cst_40
  const float v74 = 0.0f;
  // pto: %c0_index
  const int64_t v75 = 0;
  // pto: %c2_index
  const int64_t v76 = 2;
  // pto: %c512_index
  const int64_t v77 = 512;
  // pto: %c256_index
  const int64_t v78 = 256;
  // pto: %c8_index
  const int64_t v79 = 8;
  // pto: %cst_46
  const float v80 = 6.10351563E-5f;
  // pto: %cst_47
  const float v81 = 9.99999997E-7f;
  // pto: %c192_index
  const int64_t v82 = 192;
  // pto: %c64_index
  const int64_t v83 = 64;
  // pto: %c1_i32
  const int32_t v84 = 1;
  // pto: %c2_i32
  const int32_t v85 = 2;
  // pto: %c4_i32
  const int32_t v86 = 4;
  // pto: %c8_i32
  const int32_t v87 = 8;
  // pto: %c19_index
  const int64_t v88 = 19;
  // pto: %cst_55
  const float v89 = 0.25f;
  // pto: %cst_56
  const float v90 = 1.0f;
  // pto: %cst_57
  const float v91 = 4.0f;
  // pto: %cst_58
  const float v92 = 2.0f;
  // pto: %c4_index
  const int64_t v93 = 4;
  // pto: %c8192_index
  const int64_t v94 = 8192;
  // pto: %c12288_index
  const int64_t v95 = 12288;
  // pto: %246
  int64_t v96 = (int64_t) ((uint64_t) v29 * (uint64_t) v69);
  // pto: %rms_partials_inline448__ssa_v0_view
  const int64_t v97 = 1;
  // pto: %rms_partials_inline448__ssa_v0_view
  const int64_t v98 = 1;
  // pto: %rms_partials_inline448__ssa_v0_view
  const int64_t v99 = 1;
  // pto: %rms_partials_inline448__ssa_v0_view
  int64_t v100 = v96 * v65;
  // pto: %rms_partials_inline448__ssa_v0_view
  int64_t v101 = v99 * v100;
  // pto: %rms_partials_inline448__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v102 = pto::Shape<1, 1, 1, -1, -1>(v97, v98, v99, v96, v65);
  // pto: %rms_partials_inline448__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v103 = pto::Stride<-1, -1, -1, -1, -1>(v98 * v101, v101, v100, v65, v96);
  // pto: %rms_partials_inline448__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v104 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v102, v103);
  // pto: %x_flat_inline509__ssa_v0_view
  const int64_t v105 = 1;
  // pto: %x_flat_inline509__ssa_v0_view
  const int64_t v106 = 1;
  // pto: %x_flat_inline509__ssa_v0_view
  const int64_t v107 = 1;
  // pto: %x_flat_inline509__ssa_v0_view
  int64_t v108 = (int64_t) v30;
  // pto: %x_flat_inline509__ssa_v0_view
  int64_t v109 = v108 * v67;
  // pto: %x_flat_inline509__ssa_v0_view
  int64_t v110 = v107 * v109;
  // pto: %x_flat_inline509__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v111 = pto::Shape<1, 1, 1, -1, -1>(v105, v106, v107, v108, v67);
  // pto: %x_flat_inline509__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v112 = pto::Stride<-1, -1, -1, -1, -1>(v106 * v110, v110, v109, v67, v65);
  // pto: %x_flat_inline509__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v113 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v111, v112);
  // pto: %247
  int64_t v114 = (int64_t) ((uint64_t) v29 * (uint64_t) v64);
  // pto: %parts_2d_inline420__ssa_v0_view
  const int64_t v115 = 1;
  // pto: %parts_2d_inline420__ssa_v0_view
  const int64_t v116 = 1;
  // pto: %parts_2d_inline420__ssa_v0_view
  const int64_t v117 = 1;
  // pto: %parts_2d_inline420__ssa_v0_view
  int64_t v118 = v69 * v114;
  // pto: %parts_2d_inline420__ssa_v0_view
  int64_t v119 = v117 * v118;
  // pto: %parts_2d_inline420__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v120 = pto::Shape<1, 1, 1, -1, -1>(v115, v116, v117, v69, v114);
  // pto: %parts_2d_inline420__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v121 = pto::Stride<-1, -1, -1, -1, -1>(v116 * v119, v119, v118, v114, v65);
  // pto: %parts_2d_inline420__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v122 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v120, v121);
  // pto: %rms_2d_inline421__ssa_v0_view
  const int64_t v123 = 1;
  // pto: %rms_2d_inline421__ssa_v0_view
  const int64_t v124 = 1;
  // pto: %rms_2d_inline421__ssa_v0_view
  const int64_t v125 = 1;
  // pto: %rms_2d_inline421__ssa_v0_view
  int64_t v126 = (int64_t) v29;
  // pto: %rms_2d_inline421__ssa_v0_view
  int64_t v127 = v69 * v126;
  // pto: %rms_2d_inline421__ssa_v0_view
  int64_t v128 = v125 * v127;
  // pto: %rms_2d_inline421__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v129 = pto::Shape<1, 1, 1, -1, -1>(v123, v124, v125, v69, (int64_t) v29);
  // pto: %rms_2d_inline421__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v130 = pto::Stride<-1, -1, -1, -1, -1>(v124 * v128, v128, v127, v126, v65);
  // pto: %rms_2d_inline421__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v131 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v129, v130);
  // pto: %hc_base_2d_inline436__ssa_v0_view
  const int64_t v132 = 1;
  // pto: %hc_base_2d_inline436__ssa_v0_view
  const int64_t v133 = 1;
  // pto: %hc_base_2d_inline436__ssa_v0_view
  const int64_t v134 = 1;
  // pto: %hc_base_2d_inline436__ssa_v0_view
  int64_t v135 = v65 * v66;
  // pto: %hc_base_2d_inline436__ssa_v0_view
  int64_t v136 = v134 * v135;
  // pto: %hc_base_2d_inline436__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v137 = pto::Shape<1, 1, 1, -1, -1>(v132, v133, v134, v65, v66);
  // pto: %hc_base_2d_inline436__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v138 = pto::Stride<-1, -1, -1, -1, -1>(v133 * v136, v136, v135, v66, v65);
  // pto: %hc_base_2d_inline436__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v139 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v8, v137, v138);
  // pto: %comb_inline461__ssa_v0_view
  const int64_t v140 = 1;
  // pto: %comb_inline461__ssa_v0_view
  const int64_t v141 = 1;
  // pto: %comb_inline461__ssa_v0_view
  const int64_t v142 = 1;
  // pto: %comb_inline461__ssa_v0_view
  int64_t v143 = (int64_t) v30;
  // pto: %comb_inline461__ssa_v0_view
  int64_t v144 = v143 * v69;
  // pto: %comb_inline461__ssa_v0_view
  int64_t v145 = v142 * v144;
  // pto: %comb_inline461__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v146 = pto::Shape<1, 1, 1, -1, -1>(v140, v141, v142, v143, v69);
  // pto: %comb_inline461__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v147 = pto::Stride<-1, -1, -1, -1, -1>(v141 * v145, v145, v144, v69, v65);
  // pto: %comb_inline461__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v148 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v9, v146, v147);
  // pto: %hc_scale_2d_inline517__ssa_v0_view
  const int64_t v149 = 1;
  // pto: %hc_scale_2d_inline517__ssa_v0_view
  const int64_t v150 = 1;
  // pto: %hc_scale_2d_inline517__ssa_v0_view
  const int64_t v151 = 1;
  // pto: %hc_scale_2d_inline517__ssa_v0_view
  int64_t v152 = v65 * v68;
  // pto: %hc_scale_2d_inline517__ssa_v0_view
  int64_t v153 = v151 * v152;
  // pto: %hc_scale_2d_inline517__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v154 = pto::Shape<1, 1, 1, -1, -1>(v149, v150, v151, v65, v68);
  // pto: %hc_scale_2d_inline517__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v155 = pto::Stride<-1, -1, -1, -1, -1>(v150 * v153, v153, v152, v68, v65);
  // pto: %hc_scale_2d_inline517__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v156 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v10, v154, v155);
  // pto: %248
  int64_t v157 = (int64_t) ((uint64_t) v30 * (uint64_t) v93);
  // pto: %post_flat_inline459__ssa_v0_view
  const int64_t v158 = 1;
  // pto: %post_flat_inline459__ssa_v0_view
  const int64_t v159 = 1;
  // pto: %post_flat_inline459__ssa_v0_view
  const int64_t v160 = 1;
  // pto: %post_flat_inline459__ssa_v0_view
  int64_t v161 = v65 * v157;
  // pto: %post_flat_inline459__ssa_v0_view
  int64_t v162 = v160 * v161;
  // pto: %post_flat_inline459__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v163 = pto::Shape<1, 1, 1, -1, -1>(v158, v159, v160, v65, v157);
  // pto: %post_flat_inline459__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v164 = pto::Stride<-1, -1, -1, -1, -1>(v159 * v162, v162, v161, v157, v65);
  // pto: %post_flat_inline459__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v165 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v11, v163, v164);
  // pto: %mixed_view_inline481__ssa_v0_view
  const int64_t v166 = 1;
  // pto: %mixed_view_inline481__ssa_v0_view
  const int64_t v167 = 1;
  // pto: %mixed_view_inline481__ssa_v0_view
  const int64_t v168 = 1;
  // pto: %mixed_view_inline481__ssa_v0_view
  int64_t v169 = (int64_t) v30;
  // pto: %mixed_view_inline481__ssa_v0_view
  int64_t v170 = v169 * v70;
  // pto: %mixed_view_inline481__ssa_v0_view
  int64_t v171 = v168 * v170;
  // pto: %mixed_view_inline481__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v172 = pto::Shape<1, 1, 1, -1, -1>(v166, v167, v168, v169, v70);
  // pto: %mixed_view_inline481__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v173 = pto::Stride<-1, -1, -1, -1, -1>(v167 * v171, v171, v170, v70, v65);
  // pto: %mixed_view_inline481__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v174 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v12, v172, v173);
  const int32_t v175 = 0;
  auto v176 = TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>(v13, v175, v71);
  // pto: %worker_inline466__ssa_v0
  int64_t v177 = (int64_t) v31;
  // pto: %linear_workers_inline381__ssa_v0_idx
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  set_flag(PIPE_S, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
  for (int64_t v178 = v177; v178 < v14; v178 += (int64_t) v15) {
    // pto: %38, %39
    ;
    int64_t v179 = (int64_t) ((uint64_t) (v178 / v69) * (uint64_t) v72);
    // pto: %40
    ;
    int64_t v180 = v178 % v69;
    // pto: %mid_stream_rms_total_inline478__tile
    ;
    Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v181 = Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v66);
    // pto: %mid_stream_rms_total_inline478__tile
    ;
    uint64_t v182 = (uint64_t) v34;
    TASSIGN(v181, v182);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TEXPANDS(v181, v74);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    for (int64_t v183 = v75; v183 < v76; v183 += v65) {
      // pto: %41, %45, %44
      ;
      int64_t v184 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v180 * (uint64_t) v73) + (uint64_t) ((int64_t) (uint64_t) v183 * (uint64_t) v77));
      // pto: %47, %lane_inline365__ssa_v0, %46
      ;
      int64_t v185 = (int64_t) ((uint64_t) v179 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) v33) * (uint64_t) v66));
      // pto: %48
      ;
      int64_t v186 = (int64_t) ((uint64_t) v17 - (uint64_t) v185);
      // pto: %49
      ;
      int64_t v187 = v186 < v66 ? v186 : v66;
      // pto: %50
      ;
      int64_t v188 = v187 < v75 ? v75 : v187;
      // pto: %mid_stream_source_inline384__tile
      ;
      Tile<TileType::Vec, bfloat16_t, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v189 = Tile<TileType::Vec, bfloat16_t, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v188, v77);
      // pto: %mid_stream_source_inline384__tile
      ;
      uint64_t v190 = (uint64_t) v35;
      TASSIGN(v189, v190);
      // pto: %51
      ;
      int64_t v191 = v185 < v75 ? v75 : v185;
      // pto: %52
      ;
      int64_t v192 = v184 < v75 ? v75 : v184;
      // pto: %x_flat_inline509__ssa_v0_pview
      ;
      const int64_t v193 = 0;
      // pto: %x_flat_inline509__ssa_v0_pview
      ;
      __gm__ bfloat16_t* v194 = PTOAS__GLOBAL_TENSOR_DATA(v113);
      // pto: %x_flat_inline509__ssa_v0_pview
      ;
      const int64_t v195 = 1;
      // pto: %x_flat_inline509__ssa_v0_pview
      ;
      const int64_t v196 = 1;
      // pto: %x_flat_inline509__ssa_v0_pview
      ;
      const int64_t v197 = 1;
      // pto: %x_flat_inline509__ssa_v0_pview
      ;
      int64_t v198 = v188 * v67;
      // pto: %x_flat_inline509__ssa_v0_pview
      ;
      int64_t v199 = v197 * v198;
      // pto: %x_flat_inline509__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, -1> v200 = pto::Shape<1, 1, 1, -1, -1>(v195, v196, v197, v188, v77);
      // pto: %x_flat_inline509__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v201 = pto::Stride<-1, -1, -1, -1, -1>(v196 * v199, v199, v198, v67, v65);
      // pto: %x_flat_inline509__ssa_v0_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v202 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v194 + (v193 + v191 * v67 + v192 * v65), v200, v201);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v189, v202);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %mid_stream_widened_inline595__tile
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v203 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v188, v77);
      // pto: %mid_stream_widened_inline595__tile
      ;
      uint64_t v204 = (uint64_t) v36;
      TASSIGN(v203, v204);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      RoundMode v205 = RoundMode::CAST_ROUND;
      SaturationMode v206 = SaturationMode::ON;
      TCVT(v203, v189, v205, v206);
      v203.SetValidShape(v188, v77);
      // pto: %mid_stream_clean_inline382__tile
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v207 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v66, v77);
      // pto: %mid_stream_clean_inline382__tile
      ;
      uint64_t v208 = (uint64_t) v36;
      TASSIGN(v207, v208);
      pipe_barrier(PIPE_V);
      TFILLPAD<pto::TFillPadMode::InPlace>(v207, v203);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      v207.SetValidShape(v66, v77);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TPUSH<TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>, Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v176, v207);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      // pto: %0
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v209 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v66, v77);
      // pto: %0
      ;
      uint64_t v210 = (uint64_t) v36;
      TASSIGN(v209, v210);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      TMUL(v209, v207, v207);
      // pto: %tmp_tile
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v211 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v77);
      // pto: %tmp_tile
      ;
      uint64_t v212 = (uint64_t) v35;
      TASSIGN(v211, v212);
      // pto: %1
      ;
      Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v213 = Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
      // pto: %1
      ;
      uint64_t v214 = (uint64_t) v37;
      TASSIGN(v213, v214);
      pipe_barrier(PIPE_V);
      TROWSUM(v213, v209, v211);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      // pto: %mid_stream_sq_inline371__tile
      ;
      Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v215 = Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v66);
      // pto: %mid_stream_sq_inline371__tile
      ;
      uint64_t v216 = (uint64_t) v37;
      TASSIGN(v215, v216);
      // pto: %2
      ;
      Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v217 = Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v66);
      // pto: %2
      ;
      uint64_t v218 = (uint64_t) v34;
      TASSIGN(v217, v218);
      pipe_barrier(PIPE_V);
      TADD(v217, v181, v215);
    };
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %42, %43, %54, %rms_lane_inline435__ssa_v0, %53
    ;
    int64_t v219 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v180 * (uint64_t) v16) + (uint64_t) v179) + (uint64_t) ((int64_t) (uint64_t) ((int64_t) v33) * (uint64_t) v66));
    // pto: %3
    ;
    Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v220 = Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v65);
    // pto: %3
    ;
    uint64_t v221 = (uint64_t) v34;
    TASSIGN(v220, v221);
    // pto: %55
    ;
    int64_t v222 = v219 < v75 ? v75 : v219;
    // pto: %rms_partials_inline448__iter_v1_pview
    ;
    const int64_t v223 = 0;
    // pto: %rms_partials_inline448__iter_v1_pview
    ;
    __gm__ float* v224 = PTOAS__GLOBAL_TENSOR_DATA(v104);
    // pto: %rms_partials_inline448__iter_v1_pview
    ;
    const int64_t v225 = 1;
    // pto: %rms_partials_inline448__iter_v1_pview
    ;
    const int64_t v226 = 1;
    // pto: %rms_partials_inline448__iter_v1_pview
    ;
    const int64_t v227 = 1;
    // pto: %rms_partials_inline448__iter_v1_pview
    ;
    int64_t v228 = v66 * v65;
    // pto: %rms_partials_inline448__iter_v1_pview
    ;
    int64_t v229 = v227 * v228;
    // pto: %rms_partials_inline448__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 24, 1> v230 = pto::Shape<1, 1, 1, 24, 1>(v225, v226, v227, v66, v65);
    // pto: %rms_partials_inline448__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v231 = pto::Stride<-1, -1, -1, -1, -1>(v226 * v229, v229, v228, v65, v96);
    // pto: %rms_partials_inline448__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 24, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v232 = GlobalTensor<float, pto::Shape<1, 1, 1, 24, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v224 + (v223 + v222 * v65 + v75 * v96), v230, v231);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    TSTORE(v232, v220);
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
  // pto: %56, %57, %tail_lane_inline533__ssa_v0
  int64_t v233 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v177 * (uint64_t) v76) + (uint64_t) ((int64_t) v33));
  // pto: %58
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  if (v233 < v18) {
    // pto: %60
    ;
    int64_t v234 = v233 < v20 ? v233 : v20;
    // pto: %59, %61
    ;
    int64_t v235 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v233 * (uint64_t) v19) + (uint64_t) v234);
    // pto: %63
    ;
    int64_t v236 = (int64_t) ((uint64_t) v20 - (uint64_t) v233);
    // pto: %64
    ;
    int64_t v237 = v236 < v75 ? v75 : v236;
    // pto: %65
    ;
    int64_t v238 = v237 < v65 ? v237 : v65;
    // pto: %62, %66
    ;
    int64_t v239 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v235 + (uint64_t) v19) + (uint64_t) v238);
    for (int64_t v240 = v235; v240 < v239; v240 += v76) {
      // pto: %scale2_inline524__tile
      ;
      float v241 = (v5)[v76];
      // pto: %67
      ;
      int64_t v242 = (int64_t) ((uint64_t) v239 - (uint64_t) v240);
      // pto: %68
      ;
      int64_t v243 = v242 < v76 ? v242 : v76;
      // pto: %69
      ;
      int64_t v244 = (int64_t) ((uint64_t) v243 * (uint64_t) v64);
      // pto: %c_parts_load_inline432__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v245 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v244);
      // pto: %c_parts_load_inline432__ssa_v0
      ;
      uint64_t v246 = (uint64_t) v36;
      TASSIGN(v245, v246);
      // pto: %70
      ;
      int64_t v247 = (int64_t) ((uint64_t) v240 * (uint64_t) v64);
      // pto: %71
      ;
      int64_t v248 = v247 < v75 ? v75 : v247;
      // pto: %parts_2d_inline420__ssa_v0_pview
      ;
      const int64_t v249 = 0;
      // pto: %parts_2d_inline420__ssa_v0_pview
      ;
      __gm__ float* v250 = PTOAS__GLOBAL_TENSOR_DATA(v122);
      // pto: %parts_2d_inline420__ssa_v0_pview
      ;
      const int64_t v251 = 1;
      // pto: %parts_2d_inline420__ssa_v0_pview
      ;
      const int64_t v252 = 1;
      // pto: %parts_2d_inline420__ssa_v0_pview
      ;
      const int64_t v253 = 1;
      // pto: %parts_2d_inline420__ssa_v0_pview
      ;
      int64_t v254 = v69 * v114;
      // pto: %parts_2d_inline420__ssa_v0_pview
      ;
      int64_t v255 = v253 * v254;
      // pto: %parts_2d_inline420__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 16, -1> v256 = pto::Shape<1, 1, 1, 16, -1>(v251, v252, v253, v69, v244);
      // pto: %parts_2d_inline420__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v257 = pto::Stride<-1, -1, -1, -1, -1>(v252 * v255, v255, v254, v114, v65);
      // pto: %parts_2d_inline420__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v258 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v250 + (v249 + v75 * v114 + v248 * v65), v256, v257);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      TLOAD(v245, v258);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %t__tmp_v94
      ;
      Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v259 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v69, v78);
      // pto: %t__tmp_v94
      ;
      uint64_t v260 = (uint64_t) v36;
      TASSIGN(v259, v260);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      TFILLPAD<pto::TFillPadMode::InPlace>(v259, v245);
      v259.SetValidShape(v69, v78);
      // pto: %t__tmp_v95
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v261 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v78);
      // pto: %t__tmp_v95
      ;
      uint64_t v262 = (uint64_t) v36;
      TASSIGN(v261, v262);
      pipe_barrier(PIPE_V);
      TCOLSUM(v261, v259);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
      // pto: %c_total_inline444__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v263 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v79, v64);
      // pto: %c_total_inline444__ssa_v0
      ;
      uint64_t v264 = (uint64_t) v36;
      TASSIGN(v263, v264);
      // pto: %c_sq_load_inline413__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v265 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v243);
      // pto: %c_sq_load_inline413__ssa_v0
      ;
      uint64_t v266 = (uint64_t) v38;
      TASSIGN(v265, v266);
      // pto: %73
      ;
      int64_t v267 = v240 < v75 ? v75 : v240;
      // pto: %rms_2d_inline421__ssa_v0_pview
      ;
      int64_t v268 = (int64_t) v29;
      // pto: %rms_2d_inline421__ssa_v0_pview
      ;
      const int64_t v269 = 0;
      // pto: %rms_2d_inline421__ssa_v0_pview
      ;
      __gm__ float* v270 = PTOAS__GLOBAL_TENSOR_DATA(v131);
      // pto: %rms_2d_inline421__ssa_v0_pview
      ;
      const int64_t v271 = 1;
      // pto: %rms_2d_inline421__ssa_v0_pview
      ;
      const int64_t v272 = 1;
      // pto: %rms_2d_inline421__ssa_v0_pview
      ;
      const int64_t v273 = 1;
      // pto: %rms_2d_inline421__ssa_v0_pview
      ;
      int64_t v274 = v69 * v268;
      // pto: %rms_2d_inline421__ssa_v0_pview
      ;
      int64_t v275 = v273 * v274;
      // pto: %rms_2d_inline421__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 16, -1> v276 = pto::Shape<1, 1, 1, 16, -1>(v271, v272, v273, v69, v243);
      // pto: %rms_2d_inline421__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v277 = pto::Stride<-1, -1, -1, -1, -1>(v272 * v275, v275, v274, v268, v65);
      // pto: %rms_2d_inline421__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v278 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v270 + (v269 + v75 * v268 + v267 * v65), v276, v277);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
      TLOAD(v265, v278);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %t__tmp_v96
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v279 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v69, v79);
      // pto: %t__tmp_v96
      ;
      uint64_t v280 = (uint64_t) v38;
      TASSIGN(v279, v280);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      TFILLPAD<pto::TFillPadMode::InPlace>(v279, v265);
      v279.SetValidShape(v69, v79);
      // pto: %c_rms_tmp_inline376__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v281 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v79);
      // pto: %c_rms_tmp_inline376__ssa_v0
      ;
      uint64_t v282 = (uint64_t) v39;
      TASSIGN(v281, v282);
      // pto: %t__tmp_v97
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v283 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v79);
      // pto: %t__tmp_v97
      ;
      uint64_t v284 = (uint64_t) v38;
      TASSIGN(v283, v284);
      pipe_barrier(PIPE_V);
      TCOLSUM(v283, v279);
      // pto: %t__tmp_v98
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v285 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v79);
      // pto: %t__tmp_v98
      ;
      uint64_t v286 = (uint64_t) v38;
      TASSIGN(v285, v286);
      pipe_barrier(PIPE_V);
      TMULS(v285, v283, v80);
      // pto: %t__tmp_v99
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v287 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v79);
      // pto: %t__tmp_v99
      ;
      uint64_t v288 = (uint64_t) v38;
      TASSIGN(v287, v288);
      pipe_barrier(PIPE_V);
      TADDS(v287, v285, v81);
      // pto: %t__tmp_v100
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v289 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v79);
      // pto: %t__tmp_v100
      ;
      uint64_t v290 = (uint64_t) v40;
      TASSIGN(v289, v290);
      pipe_barrier(PIPE_V);
      TRSQRT(v289, v287, v281);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      // pto: %c_inv_inline499__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v291 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v79, v65);
      // pto: %c_inv_inline499__ssa_v0
      ;
      uint64_t v292 = (uint64_t) v40;
      TASSIGN(v291, v292);
      // pto: %comb_inv_inline587__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v293 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v79, v65);
      // pto: %comb_inv_inline587__ssa_v0
      ;
      uint64_t v294 = (uint64_t) v40;
      TASSIGN(v293, v294);
      // pto: %t__tmp_v101
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v295 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v79, v69);
      // pto: %t__tmp_v101
      ;
      uint64_t v296 = (uint64_t) v41;
      TASSIGN(v295, v296);
      // pto: %slice_view
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 16, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v297;
      // pto: %slice_view
      ;
      uint64_t v298 = (uint64_t) v41;
      TASSIGN(v297, v298);
      // pto: %raw_inline411__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v299 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v79, v69);
      // pto: %raw_inline411__ssa_v0
      ;
      uint64_t v300 = (uint64_t) v36;
      TASSIGN(v299, v300);
      TADDS(v299, v297, v74);
      // pto: %base_inline510__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v301 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %base_inline510__ssa_v0
      ;
      uint64_t v302 = (uint64_t) v38;
      TASSIGN(v301, v302);
      // pto: %hc_base_2d_inline436__ssa_v0_pview
      ;
      __gm__ float* v303 = PTOAS__GLOBAL_TENSOR_DATA(v139);
      // pto: %hc_base_2d_inline436__ssa_v0_pview
      ;
      unsigned v304 = 8;
      // pto: %hc_base_2d_inline436__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 16> v305 = pto::Shape<1, 1, 1, 1, 16>();
      // pto: %hc_base_2d_inline436__ssa_v0_pview
      ;
      pto::Stride<24, 24, 24, 24, 1> v306 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %hc_base_2d_inline436__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v307 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v303 + v304, v305, v306);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      TLOAD(v301, v307);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      // pto: %t__tmp_v102
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v308 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v79, v69);
      // pto: %t__tmp_v102
      ;
      uint64_t v309 = (uint64_t) v42;
      TASSIGN(v308, v309);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v308, v299, v293);
      // pto: %t__tmp_v103
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v310 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v79, v69);
      // pto: %t__tmp_v103
      ;
      uint64_t v311 = (uint64_t) v42;
      TASSIGN(v310, v311);
      pipe_barrier(PIPE_V);
      TMULS(v310, v308, v241);
      // pto: %t__tmp_v104
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v312 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v79, v69);
      // pto: %t__tmp_v104
      ;
      uint64_t v313 = (uint64_t) v36;
      TASSIGN(v312, v313);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      TCOLEXPAND(v312, v301);
      // pto: %logits_inline520__ssa_v1
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v314 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v79, v69);
      // pto: %logits_inline520__ssa_v1
      ;
      uint64_t v315 = (uint64_t) v43;
      TASSIGN(v314, v315);
      pipe_barrier(PIPE_V);
      TADD(v314, v310, v312);
      // pto: %t__tmp_v105
      ;
      Tile<TileType::Vec, float, 2, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v316 = Tile<TileType::Vec, float, 2, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v69);
      // pto: %t__tmp_v105
      ;
      uint64_t v317 = (uint64_t) v43;
      TASSIGN(v316, v317);
      // pto: %regular_logits_inline571__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v318 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %regular_logits_inline571__ssa_v0
      ;
      uint64_t v319 = (uint64_t) v43;
      TASSIGN(v318, v319);
      // pto: %ids_inline523__ci_tmp_v0
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v320 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %ids_inline523__ci_tmp_v0
      ;
      uint64_t v321 = (uint64_t) v36;
      TASSIGN(v320, v321);
      // pto: %ids_inline523__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v322 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %ids_inline523__ssa_v1
      ;
      uint64_t v323 = (uint64_t) v44;
      TASSIGN(v322, v323);
      // pto: %ci_tmp_view
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v324;
      TRESHAPE(v324, v320);
      // pto: %ci_dst_view
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v325;
      TRESHAPE(v325, v322);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v325, v71, v324);
      // pto: %tmp_inline429__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v326 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %tmp_inline429__ssa_v1
      ;
      uint64_t v327 = (uint64_t) v36;
      TASSIGN(v326, v327);
      // pto: %ids16_inline506__ssa_v1
      ;
      Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v328 = Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v83);
      // pto: %ids16_inline506__ssa_v1
      ;
      uint64_t v329 = (uint64_t) v44;
      TASSIGN(v328, v329);
      // pto: %xor_tmp16_inline532__ssa_v1
      ;
      Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v330 = Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v83);
      // pto: %xor_tmp16_inline532__ssa_v1
      ;
      uint64_t v331 = (uint64_t) v45;
      TASSIGN(v330, v331);
      // pto: %t__tmp_v106
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v332 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v106
      ;
      uint64_t v333 = (uint64_t) v46;
      TASSIGN(v332, v333);
      pipe_barrier(PIPE_V);
      TEXPANDS(v332, v84);
      // pto: %ri1_mask_inline534__ssa_v1
      ;
      Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v334 = Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v83);
      // pto: %ri1_mask_inline534__ssa_v1
      ;
      uint64_t v335 = (uint64_t) v46;
      TASSIGN(v334, v335);
      // pto: %t__tmp_v107
      ;
      Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v336 = Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v83);
      // pto: %t__tmp_v107
      ;
      uint64_t v337 = (uint64_t) v47;
      TASSIGN(v336, v337);
      pipe_barrier(PIPE_V);
      TXOR(v336, v328, v334, v330);
      // pto: %ri1_inline537__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v338 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %ri1_inline537__ssa_v1
      ;
      uint64_t v339 = (uint64_t) v47;
      TASSIGN(v338, v339);
      // pto: %t__tmp_v108
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v340 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v108
      ;
      uint64_t v341 = (uint64_t) v46;
      TASSIGN(v340, v341);
      pipe_barrier(PIPE_V);
      TEXPANDS(v340, v85);
      // pto: %ri2_mask_inline574__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v342 = Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v83);
      // pto: %ri2_mask_inline574__ssa_v0
      ;
      uint64_t v343 = (uint64_t) v46;
      TASSIGN(v342, v343);
      // pto: %t__tmp_v109
      ;
      Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v344 = Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v83);
      // pto: %t__tmp_v109
      ;
      uint64_t v345 = (uint64_t) v42;
      TASSIGN(v344, v345);
      pipe_barrier(PIPE_V);
      TXOR(v344, v328, v342, v330);
      // pto: %ri2_inline405__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v346 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %ri2_inline405__ssa_v0
      ;
      uint64_t v347 = (uint64_t) v42;
      TASSIGN(v346, v347);
      // pto: %t__tmp_v110
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v348 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v110
      ;
      uint64_t v349 = (uint64_t) v46;
      TASSIGN(v348, v349);
      pipe_barrier(PIPE_V);
      TEXPANDS(v348, v86);
      // pto: %ci1_mask_inline575__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v350 = Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v83);
      // pto: %ci1_mask_inline575__ssa_v0
      ;
      uint64_t v351 = (uint64_t) v46;
      TASSIGN(v350, v351);
      // pto: %t__tmp_v111
      ;
      Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v352 = Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v83);
      // pto: %t__tmp_v111
      ;
      uint64_t v353 = (uint64_t) v48;
      TASSIGN(v352, v353);
      pipe_barrier(PIPE_V);
      TXOR(v352, v328, v350, v330);
      // pto: %ci1_inline480__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v354 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %ci1_inline480__ssa_v0
      ;
      uint64_t v355 = (uint64_t) v48;
      TASSIGN(v354, v355);
      // pto: %t__tmp_v112
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v356 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v112
      ;
      uint64_t v357 = (uint64_t) v46;
      TASSIGN(v356, v357);
      pipe_barrier(PIPE_V);
      TEXPANDS(v356, v87);
      // pto: %ci2_mask_inline577__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v358 = Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v83);
      // pto: %ci2_mask_inline577__ssa_v0
      ;
      uint64_t v359 = (uint64_t) v46;
      TASSIGN(v358, v359);
      // pto: %t__tmp_v113
      ;
      Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v360 = Tile<TileType::Vec, int16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v83);
      // pto: %t__tmp_v113
      ;
      uint64_t v361 = (uint64_t) v49;
      TASSIGN(v360, v361);
      pipe_barrier(PIPE_V);
      TXOR(v360, v328, v358, v330);
      // pto: %ci2_inline545__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v362 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %ci2_inline545__ssa_v0
      ;
      uint64_t v363 = (uint64_t) v49;
      TASSIGN(v362, v363);
      // pto: %t__tmp_v114
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v364 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v114
      ;
      uint64_t v365 = (uint64_t) v45;
      TASSIGN(v364, v365);
      pipe_barrier(PIPE_V);
      TGATHER(v364, v318, v338, v326);
      // pto: %max2_inline548__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v366 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %max2_inline548__ssa_v1
      ;
      uint64_t v367 = (uint64_t) v45;
      TASSIGN(v366, v367);
      pipe_barrier(PIPE_V);
      TMAX(v366, v318, v364);
      // pto: %t__tmp_v115
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v368 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v115
      ;
      uint64_t v369 = (uint64_t) v46;
      TASSIGN(v368, v369);
      pipe_barrier(PIPE_V);
      TGATHER(v368, v366, v346, v326);
      // pto: %max4_inline400__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v370 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %max4_inline400__ssa_v1
      ;
      uint64_t v371 = (uint64_t) v45;
      TASSIGN(v370, v371);
      pipe_barrier(PIPE_V);
      TMAX(v370, v366, v368);
      // pto: %t__tmp_v116
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v372 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v116
      ;
      uint64_t v373 = (uint64_t) v45;
      TASSIGN(v372, v373);
      pipe_barrier(PIPE_V);
      TSUB(v372, v318, v370);
      // pto: %exps_inline552__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v374 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %exps_inline552__ssa_v1
      ;
      uint64_t v375 = (uint64_t) v45;
      TASSIGN(v374, v375);
      pipe_barrier(PIPE_V);
      TEXP(v374, v372);
      // pto: %t__tmp_v117
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v376 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v117
      ;
      uint64_t v377 = (uint64_t) v46;
      TASSIGN(v376, v377);
      pipe_barrier(PIPE_V);
      TGATHER(v376, v374, v338, v326);
      // pto: %sum2_inline623__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v378 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %sum2_inline623__ssa_v1
      ;
      uint64_t v379 = (uint64_t) v46;
      TASSIGN(v378, v379);
      pipe_barrier(PIPE_V);
      TADD(v378, v374, v376);
      // pto: %t__tmp_v118
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v380 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v118
      ;
      uint64_t v381 = (uint64_t) v43;
      TASSIGN(v380, v381);
      pipe_barrier(PIPE_V);
      TGATHER(v380, v378, v346, v326);
      // pto: %sum4_inline554__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v382 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %sum4_inline554__ssa_v1
      ;
      uint64_t v383 = (uint64_t) v46;
      TASSIGN(v382, v383);
      pipe_barrier(PIPE_V);
      TADD(v382, v378, v380);
      // pto: %t__tmp_v119
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v384 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v119
      ;
      uint64_t v385 = (uint64_t) v45;
      TASSIGN(v384, v385);
      pipe_barrier(PIPE_V);
      TDIV(v384, v374, v382);
      // pto: %probs_inline447__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v386 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %probs_inline447__ssa_v1
      ;
      uint64_t v387 = (uint64_t) v45;
      TASSIGN(v386, v387);
      pipe_barrier(PIPE_V);
      TADDS(v386, v384, v81);
      // pto: %t__tmp_v120
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v388 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v120
      ;
      uint64_t v389 = (uint64_t) v46;
      TASSIGN(v388, v389);
      pipe_barrier(PIPE_V);
      TGATHER(v388, v386, v354, v326);
      // pto: %col2_inline558__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v390 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %col2_inline558__ssa_v4
      ;
      uint64_t v391 = (uint64_t) v46;
      TASSIGN(v390, v391);
      pipe_barrier(PIPE_V);
      TADD(v390, v386, v388);
      // pto: %t__tmp_v121
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v392 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v121
      ;
      uint64_t v393 = (uint64_t) v43;
      TASSIGN(v392, v393);
      pipe_barrier(PIPE_V);
      TGATHER(v392, v390, v362, v326);
      // pto: %t__tmp_v122
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v394 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v122
      ;
      uint64_t v395 = (uint64_t) v46;
      TASSIGN(v394, v395);
      pipe_barrier(PIPE_V);
      TADD(v394, v390, v392);
      // pto: %col4_inline410__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v396 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %col4_inline410__ssa_v4
      ;
      uint64_t v397 = (uint64_t) v46;
      TASSIGN(v396, v397);
      pipe_barrier(PIPE_V);
      TADDS(v396, v394, v81);
      // pto: %cur_inline561__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v398 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %cur_inline561__ssa_v4
      ;
      uint64_t v399 = (uint64_t) v45;
      TASSIGN(v398, v399);
      pipe_barrier(PIPE_V);
      TDIV(v398, v386, v396);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      for (int64_t v400 = v75; v400 < v88; v400 += v65) {
        // pto: %t__tmp_v123
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v401 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
        // pto: %t__tmp_v123
        ;
        uint64_t v402 = (uint64_t) v46;
        TASSIGN(v401, v402);
        pipe_barrier(PIPE_V);
        TGATHER(v401, v398, v338, v326);
        // pto: %row2_inline494__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v403 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
        // pto: %row2_inline494__ssa_v1
        ;
        uint64_t v404 = (uint64_t) v46;
        TASSIGN(v403, v404);
        pipe_barrier(PIPE_V);
        TADD(v403, v398, v401);
        // pto: %t__tmp_v124
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v405 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
        // pto: %t__tmp_v124
        ;
        uint64_t v406 = (uint64_t) v43;
        TASSIGN(v405, v406);
        pipe_barrier(PIPE_V);
        TGATHER(v405, v403, v346, v326);
        // pto: %t__tmp_v125
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v407 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
        // pto: %t__tmp_v125
        ;
        uint64_t v408 = (uint64_t) v46;
        TASSIGN(v407, v408);
        pipe_barrier(PIPE_V);
        TADD(v407, v403, v405);
        // pto: %row4_inline631__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v409 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
        // pto: %row4_inline631__ssa_v1
        ;
        uint64_t v410 = (uint64_t) v46;
        TASSIGN(v409, v410);
        pipe_barrier(PIPE_V);
        TADDS(v409, v407, v81);
        // pto: %norm_inline442__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v411 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
        // pto: %norm_inline442__ssa_v1
        ;
        uint64_t v412 = (uint64_t) v46;
        TASSIGN(v411, v412);
        pipe_barrier(PIPE_V);
        TDIV(v411, v398, v409);
        // pto: %t__tmp_v126
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v413 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
        // pto: %t__tmp_v126
        ;
        uint64_t v414 = (uint64_t) v43;
        TASSIGN(v413, v414);
        pipe_barrier(PIPE_V);
        TGATHER(v413, v411, v354, v326);
        // pto: %col2_inline558__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v415 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
        // pto: %col2_inline558__ssa_v7
        ;
        uint64_t v416 = (uint64_t) v43;
        TASSIGN(v415, v416);
        pipe_barrier(PIPE_V);
        TADD(v415, v411, v413);
        // pto: %t__tmp_v127
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v417 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
        // pto: %t__tmp_v127
        ;
        uint64_t v418 = (uint64_t) v50;
        TASSIGN(v417, v418);
        pipe_barrier(PIPE_V);
        TGATHER(v417, v415, v362, v326);
        // pto: %t__tmp_v128
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v419 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
        // pto: %t__tmp_v128
        ;
        uint64_t v420 = (uint64_t) v43;
        TASSIGN(v419, v420);
        pipe_barrier(PIPE_V);
        TADD(v419, v415, v417);
        // pto: %col4_inline410__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v421 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
        // pto: %col4_inline410__ssa_v7
        ;
        uint64_t v422 = (uint64_t) v43;
        TASSIGN(v421, v422);
        pipe_barrier(PIPE_V);
        TADDS(v421, v419, v81);
        // pto: %cur_inline561__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v423 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
        // pto: %cur_inline561__ssa_v7
        ;
        uint64_t v424 = (uint64_t) v45;
        TASSIGN(v423, v424);
        pipe_barrier(PIPE_V);
        TDIV(v423, v411, v421);
      };
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      // pto: %t__tmp_v129
      ;
      Tile<TileType::Vec, float, 2, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v425 = Tile<TileType::Vec, float, 2, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v69);
      // pto: %t__tmp_v129
      ;
      uint64_t v426 = (uint64_t) v45;
      TASSIGN(v425, v426);
      v425.SetValidShape(v243, v69);
      // pto: %comb_inline461__ssa_v0_pview
      ;
      const int64_t v427 = 0;
      // pto: %comb_inline461__ssa_v0_pview
      ;
      __gm__ float* v428 = PTOAS__GLOBAL_TENSOR_DATA(v148);
      // pto: %comb_inline461__ssa_v0_pview
      ;
      const int64_t v429 = 1;
      // pto: %comb_inline461__ssa_v0_pview
      ;
      const int64_t v430 = 1;
      // pto: %comb_inline461__ssa_v0_pview
      ;
      const int64_t v431 = 1;
      // pto: %comb_inline461__ssa_v0_pview
      ;
      int64_t v432 = v243 * v69;
      // pto: %comb_inline461__ssa_v0_pview
      ;
      int64_t v433 = v431 * v432;
      // pto: %comb_inline461__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, 16> v434 = pto::Shape<1, 1, 1, -1, 16>(v429, v430, v431, v243, v69);
      // pto: %comb_inline461__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v435 = pto::Stride<-1, -1, -1, -1, -1>(v430 * v433, v433, v432, v69, v65);
      // pto: %comb_inline461__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v436 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v428 + (v427 + v267 * v69 + v75 * v65), v434, v435);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      TSTORE(v436, v425);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    };
  } else {
    // pto: %76
    ;
    int64_t v437 = (int64_t) ((uint64_t) v233 - (uint64_t) v18);
    // pto: %78
    ;
    int64_t v438 = v437 < v22 ? v437 : v22;
    // pto: %77, %79, %80
    ;
    int64_t v439 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v437 * (uint64_t) v21) + (uint64_t) v438) * (uint64_t) v76);
    // pto: %81
    ;
    int64_t v440 = (int64_t) ((uint64_t) v22 - (uint64_t) v437);
    // pto: %82
    ;
    int64_t v441 = v440 < v75 ? v75 : v440;
    // pto: %83
    ;
    int64_t v442 = v441 < v65 ? v441 : v65;
    // pto: %86, %84, %85
    ;
    int64_t v443 = (int64_t) ((uint64_t) v439 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v21 + (uint64_t) v442) * (uint64_t) v76));
    // pto: %87
    ;
    // pto: %88
    ;
    // pto: %d_begin_inline611__phi_v2
    ;
    int64_t v444;
    // pto: %d_end_inline613__phi_v2
    ;
    int64_t v445;
    // pto: %work_begin_inline406__phi_v3
    ;
    int64_t v446;
    // pto: %work_end_inline601__phi_v3
    ;
    int64_t v447;
    if (v24 < v25) {
      // pto: %89
      ;
      // pto: %d_worker_inline632__phi_v2
      ;
      int64_t v448;
      // pto: %d_workers_inline369__phi_v2
      ;
      int64_t v449;
      // pto: %pair_id_inline615__phi_v2
      ;
      int64_t v450;
      if (v437 < v26) {
        // pto: %90
        ;
        int64_t v451 = (int64_t) ((uint64_t) v27 + (uint64_t) v65);
        // pto: %91
        ;
        // pto: %93
        ;
        v448 = v437 % v451;
        v449 = v451;
        v450 = v437 / v451;
      } else {
        // pto: %95
        ;
        int64_t v452 = (int64_t) ((uint64_t) v437 - (uint64_t) v26);
        // pto: %97, %96
        ;
        // pto: %98
        ;
        v448 = v452 % v27;
        v449 = v27;
        v450 = (int64_t) ((uint64_t) v28 + (uint64_t) (v452 / v27));
      };
      // pto: %99
      ;
      uint64_t v453 = (uint64_t) v450;
      // pto: %99
      ;
      int64_t v454 = (int64_t) (v453 * (uint64_t) v76);
      // pto: %100
      ;
      int64_t v455 = v23 < v454 ? v23 : v454;
      // pto: %101
      ;
      int64_t v456 = (int64_t) ((uint64_t) v455 + (uint64_t) v76);
      // pto: %102
      ;
      // pto: %103
      ;
      int64_t v457 = v79 / v449;
      // pto: %104
      ;
      int64_t v458 = v79 % v449;
      // pto: %105
      ;
      uint64_t v459 = (uint64_t) v448;
      // pto: %106
      ;
      bool v460 = v448 < v458;
      // pto: %106
      ;
      int64_t v461 = v460 ? v448 : v458;
      // pto: %105, %107
      ;
      int64_t v462 = (int64_t) ((uint64_t) ((int64_t) v459 * (uint64_t) v457) + (uint64_t) v461);
      // pto: %109
      ;
      uint64_t v463 = (uint64_t) v448;
      // pto: %109
      ;
      int64_t v464 = (int64_t) ((uint64_t) v458 - v463);
      // pto: %110
      ;
      int64_t v465 = v464 < v75 ? v75 : v464;
      // pto: %111
      ;
      int64_t v466 = v465 < v65 ? v465 : v65;
      // pto: %108, %112
      ;
      v444 = v462;
      v445 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v462 + (uint64_t) v457) + (uint64_t) v466);
      v446 = v455;
      v447 = v23 < v456 ? v23 : v456;
    } else {
      v444 = v75;
      v445 = v79;
      v446 = v439;
      v447 = v23 < v443 ? v23 : v443;
    };
    // pto: %113
    ;
    bool v467 = v25 <= v437;
    // pto: %work_begin_inline406__phi_v5
    ;
    int64_t v468 = v467 ? v75 : v446;
    // pto: %work_end_inline601__phi_v5
    ;
    int64_t v469 = v467 ? v75 : v447;
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    for (int64_t v470 = v468; v470 < v469; v470 += v76) {
      // pto: %114
      ;
      int64_t v471 = (int64_t) ((uint64_t) v469 - (uint64_t) v470);
      // pto: %115
      ;
      int64_t v472 = v471 < v76 ? v471 : v76;
      // pto: %116
      ;
      int64_t v473 = (int64_t) ((uint64_t) v472 * (uint64_t) v64);
      // pto: %m_parts_load_inline617__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v474 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v473);
      // pto: %m_parts_load_inline617__ssa_v0
      ;
      uint64_t v475 = (uint64_t) v36;
      TASSIGN(v474, v475);
      // pto: %117
      ;
      int64_t v476 = (int64_t) ((uint64_t) v470 * (uint64_t) v64);
      // pto: %118
      ;
      int64_t v477 = v476 < v75 ? v75 : v476;
      // pto: %120
      ;
      const int64_t v478 = 0;
      // pto: %120
      ;
      __gm__ float* v479 = PTOAS__GLOBAL_TENSOR_DATA(v122);
      // pto: %120
      ;
      const int64_t v480 = 1;
      // pto: %120
      ;
      const int64_t v481 = 1;
      // pto: %120
      ;
      const int64_t v482 = 1;
      // pto: %120
      ;
      int64_t v483 = v69 * v114;
      // pto: %120
      ;
      int64_t v484 = v482 * v483;
      // pto: %120
      ;
      pto::Shape<1, 1, 1, 16, -1> v485 = pto::Shape<1, 1, 1, 16, -1>(v480, v481, v482, v69, v473);
      // pto: %120
      ;
      pto::Stride<-1, -1, -1, -1, -1> v486 = pto::Stride<-1, -1, -1, -1, -1>(v481 * v484, v484, v483, v114, v65);
      // pto: %120
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v487 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v479 + (v478 + v75 * v114 + v477 * v65), v485, v486);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      TLOAD(v474, v487);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      // pto: %t__tmp_v130
      ;
      Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v488 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v69, v78);
      // pto: %t__tmp_v130
      ;
      uint64_t v489 = (uint64_t) v36;
      TASSIGN(v488, v489);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      TFILLPAD<pto::TFillPadMode::InPlace>(v488, v474);
      v488.SetValidShape(v69, v78);
      // pto: %t__tmp_v131
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v490 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v78);
      // pto: %t__tmp_v131
      ;
      uint64_t v491 = (uint64_t) v36;
      TASSIGN(v490, v491);
      pipe_barrier(PIPE_V);
      TCOLSUM(v490, v488);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      // pto: %m_total_inline618__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v492 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v79, v64);
      // pto: %m_total_inline618__ssa_v0
      ;
      uint64_t v493 = (uint64_t) v36;
      TASSIGN(v492, v493);
      // pto: %m_sq_load_inline620__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v494 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v472);
      // pto: %m_sq_load_inline620__ssa_v0
      ;
      uint64_t v495 = (uint64_t) v38;
      TASSIGN(v494, v495);
      // pto: %121
      ;
      int64_t v496 = v470 < v75 ? v75 : v470;
      // pto: %122
      ;
      int64_t v497 = (int64_t) v29;
      // pto: %122
      ;
      const int64_t v498 = 0;
      // pto: %122
      ;
      __gm__ float* v499 = PTOAS__GLOBAL_TENSOR_DATA(v131);
      // pto: %122
      ;
      const int64_t v500 = 1;
      // pto: %122
      ;
      const int64_t v501 = 1;
      // pto: %122
      ;
      const int64_t v502 = 1;
      // pto: %122
      ;
      int64_t v503 = v69 * v497;
      // pto: %122
      ;
      int64_t v504 = v502 * v503;
      // pto: %122
      ;
      pto::Shape<1, 1, 1, 16, -1> v505 = pto::Shape<1, 1, 1, 16, -1>(v500, v501, v502, v69, v472);
      // pto: %122
      ;
      pto::Stride<-1, -1, -1, -1, -1> v506 = pto::Stride<-1, -1, -1, -1, -1>(v501 * v504, v504, v503, v497, v65);
      // pto: %122
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v507 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v499 + (v498 + v75 * v497 + v496 * v65), v505, v506);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      TLOAD(v494, v507);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      // pto: %t__tmp_v132
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v508 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v69, v79);
      // pto: %t__tmp_v132
      ;
      uint64_t v509 = (uint64_t) v38;
      TASSIGN(v508, v509);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      TFILLPAD<pto::TFillPadMode::InPlace>(v508, v494);
      v508.SetValidShape(v69, v79);
      // pto: %m_rms_tmp_inline415__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v510 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v79);
      // pto: %m_rms_tmp_inline415__ssa_v0
      ;
      uint64_t v511 = (uint64_t) v39;
      TASSIGN(v510, v511);
      // pto: %t__tmp_v133
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v512 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v79);
      // pto: %t__tmp_v133
      ;
      uint64_t v513 = (uint64_t) v38;
      TASSIGN(v512, v513);
      pipe_barrier(PIPE_V);
      TCOLSUM(v512, v508);
      // pto: %t__tmp_v134
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v514 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v79);
      // pto: %t__tmp_v134
      ;
      uint64_t v515 = (uint64_t) v38;
      TASSIGN(v514, v515);
      pipe_barrier(PIPE_V);
      TMULS(v514, v512, v80);
      // pto: %t__tmp_v135
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v516 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v79);
      // pto: %t__tmp_v135
      ;
      uint64_t v517 = (uint64_t) v38;
      TASSIGN(v516, v517);
      pipe_barrier(PIPE_V);
      TADDS(v516, v514, v81);
      // pto: %t__tmp_v136
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v518 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v79);
      // pto: %t__tmp_v136
      ;
      uint64_t v519 = (uint64_t) v51;
      TASSIGN(v518, v519);
      pipe_barrier(PIPE_V);
      TRSQRT(v518, v516, v510);
      // pto: %m_inv_inline621__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v520 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v79, v65);
      // pto: %m_inv_inline621__ssa_v0
      ;
      uint64_t v521 = (uint64_t) v51;
      TASSIGN(v520, v521);
      // pto: %gate_base_inline498__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v522 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v79);
      // pto: %gate_base_inline498__ssa_v0
      ;
      uint64_t v523 = (uint64_t) v52;
      TASSIGN(v522, v523);
      // pto: %123
      ;
      __gm__ float* v524 = PTOAS__GLOBAL_TENSOR_DATA(v139);
      // pto: %123
      ;
      pto::Shape<1, 1, 1, 1, 8> v525 = pto::Shape<1, 1, 1, 1, 8>();
      // pto: %123
      ;
      pto::Stride<24, 24, 24, 24, 1> v526 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %123
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v527 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v524, v525, v526);
      TLOAD(v522, v527);
      // pto: %gate_scales_inline624__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v528 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v68);
      // pto: %gate_scales_inline624__ssa_v0
      ;
      uint64_t v529 = (uint64_t) v53;
      TASSIGN(v528, v529);
      // pto: %hc_scale_2d_inline517__ssa_v0_pview
      ;
      __gm__ float* v530 = PTOAS__GLOBAL_TENSOR_DATA(v156);
      // pto: %hc_scale_2d_inline517__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 3> v531 = pto::Shape<1, 1, 1, 1, 3>();
      // pto: %hc_scale_2d_inline517__ssa_v0_pview
      ;
      pto::Stride<3, 3, 3, 3, 1> v532 = pto::Stride<3, 3, 3, 3, 1>();
      // pto: %hc_scale_2d_inline517__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND> v533 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND>(v530, v531, v532);
      TLOAD(v528, v533);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      // pto: %t__ci_tmp_v1
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v534 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__ci_tmp_v1
      ;
      uint64_t v535 = (uint64_t) v38;
      TASSIGN(v534, v535);
      // pto: %t__tmp_v137
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v536 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v79);
      // pto: %t__tmp_v137
      ;
      uint64_t v537 = (uint64_t) v54;
      TASSIGN(v536, v537);
      // pto: %124
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v538;
      TRESHAPE(v538, v534);
      // pto: %125
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v539;
      TRESHAPE(v539, v536);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v539, v71, v538);
      // pto: %gate_ids_inline626__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v540 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v79);
      // pto: %gate_ids_inline626__ssa_v0
      ;
      uint64_t v541 = (uint64_t) v38;
      TASSIGN(v540, v541);
      pipe_barrier(PIPE_V);
      RoundMode v542 = RoundMode::CAST_ROUND;
      SaturationMode v543 = SaturationMode::ON;
      TCVT(v540, v536, v542, v543);
      // pto: %t__tmp_v138
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v544 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v79);
      // pto: %t__tmp_v138
      ;
      uint64_t v545 = (uint64_t) v38;
      TASSIGN(v544, v545);
      pipe_barrier(PIPE_V);
      TMULS(v544, v540, v89);
      // pto: %gate_scale_ids_inline458__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v546 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v79);
      // pto: %gate_scale_ids_inline458__ssa_v0
      ;
      uint64_t v547 = (uint64_t) v38;
      TASSIGN(v546, v547);
      pipe_barrier(PIPE_V);
      RoundMode v548 = RoundMode::CAST_FLOOR;
      SaturationMode v549 = SaturationMode::ON;
      TCVT(v546, v544, v548, v549);
      // pto: %gate_tmp_inline362__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v550 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v79);
      // pto: %gate_tmp_inline362__ssa_v0
      ;
      uint64_t v551 = (uint64_t) v55;
      TASSIGN(v550, v551);
      // pto: %gate_scale_vec_inline629__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v552 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v79);
      // pto: %gate_scale_vec_inline629__ssa_v0
      ;
      uint64_t v553 = (uint64_t) v40;
      TASSIGN(v552, v553);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      TGATHER(v552, v528, v546, v550);
      // pto: %t__tmp_v139
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v554 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v79, v79);
      // pto: %t__tmp_v139
      ;
      uint64_t v555 = (uint64_t) v36;
      TASSIGN(v554, v555);
      // pto: %126
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v556;
      // pto: %126
      ;
      uint64_t v557 = (uint64_t) v36;
      TASSIGN(v556, v557);
      // pto: %gate_raw_inline375__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v558 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v79, v79);
      // pto: %gate_raw_inline375__ssa_v0
      ;
      uint64_t v559 = (uint64_t) v36;
      TASSIGN(v558, v559);
      TADDS(v558, v556, v74);
      // pto: %gate_norm_inline630__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v560 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v79, v79);
      // pto: %gate_norm_inline630__ssa_v0
      ;
      uint64_t v561 = (uint64_t) v36;
      TASSIGN(v560, v561);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v560, v558, v520);
      // pto: %gate_scaled_inline591__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v562 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v79, v79);
      // pto: %gate_scaled_inline591__ssa_v0
      ;
      uint64_t v563 = (uint64_t) v36;
      TASSIGN(v562, v563);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v562, v560, v552);
      // pto: %t__tmp_v140
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v564 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v79, v79);
      // pto: %t__tmp_v140
      ;
      uint64_t v565 = (uint64_t) v46;
      TASSIGN(v564, v565);
      TCOLEXPAND(v564, v522);
      // pto: %gate_logits_inline633__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v566 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v79, v79);
      // pto: %gate_logits_inline633__ssa_v0
      ;
      uint64_t v567 = (uint64_t) v36;
      TASSIGN(v566, v567);
      pipe_barrier(PIPE_V);
      TADD(v566, v562, v564);
      // pto: %t__tmp_v141
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v568 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v79, v79);
      // pto: %t__tmp_v141
      ;
      uint64_t v569 = (uint64_t) v36;
      TASSIGN(v568, v569);
      pipe_barrier(PIPE_V);
      TNEG(v568, v566);
      // pto: %t__tmp_v142
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v570 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v79, v79);
      // pto: %t__tmp_v142
      ;
      uint64_t v571 = (uint64_t) v36;
      TASSIGN(v570, v571);
      pipe_barrier(PIPE_V);
      TEXP(v570, v568);
      // pto: %t__tmp_v143
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v572 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v79, v79);
      // pto: %t__tmp_v143
      ;
      uint64_t v573 = (uint64_t) v36;
      TASSIGN(v572, v573);
      pipe_barrier(PIPE_V);
      TADDS(v572, v570, v90);
      // pto: %gate_sigmoid_inline634__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v574 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v79, v79);
      // pto: %gate_sigmoid_inline634__ssa_v0
      ;
      uint64_t v575 = (uint64_t) v56;
      TASSIGN(v574, v575);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
      TRECIP(v574, v572);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      // pto: %t__ci_tmp_v2
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v576 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__ci_tmp_v2
      ;
      uint64_t v577 = (uint64_t) v36;
      TASSIGN(v576, v577);
      // pto: %t__tmp_v144
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v578 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v144
      ;
      uint64_t v579 = (uint64_t) v43;
      TASSIGN(v578, v579);
      // pto: %127
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v580;
      TRESHAPE(v580, v576);
      // pto: %128
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v581;
      TRESHAPE(v581, v578);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v581, v71, v580);
      // pto: %post_ids_float_inline635__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v582 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %post_ids_float_inline635__ssa_v0
      ;
      uint64_t v583 = (uint64_t) v36;
      TASSIGN(v582, v583);
      pipe_barrier(PIPE_V);
      RoundMode v584 = RoundMode::CAST_ROUND;
      SaturationMode v585 = SaturationMode::ON;
      TCVT(v582, v578, v584, v585);
      // pto: %t__tmp_v145
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v586 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v145
      ;
      uint64_t v587 = (uint64_t) v45;
      TASSIGN(v586, v587);
      pipe_barrier(PIPE_V);
      TMULS(v586, v582, v89);
      // pto: %t__tmp_v146
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v588 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v146
      ;
      uint64_t v589 = (uint64_t) v45;
      TASSIGN(v588, v589);
      pipe_barrier(PIPE_V);
      RoundMode v590 = RoundMode::CAST_FLOOR;
      SaturationMode v591 = SaturationMode::ON;
      TCVT(v588, v586, v590, v591);
      // pto: %post_rows_inline392__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v592 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %post_rows_inline392__ssa_v0
      ;
      uint64_t v593 = (uint64_t) v45;
      TASSIGN(v592, v593);
      pipe_barrier(PIPE_V);
      RoundMode v594 = RoundMode::CAST_ROUND;
      SaturationMode v595 = SaturationMode::ON;
      TCVT(v592, v588, v594, v595);
      // pto: %t__tmp_v147
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v596 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v147
      ;
      uint64_t v597 = (uint64_t) v45;
      TASSIGN(v596, v597);
      pipe_barrier(PIPE_V);
      TMULS(v596, v592, v91);
      // pto: %t__tmp_v148
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v598 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v148
      ;
      uint64_t v599 = (uint64_t) v36;
      TASSIGN(v598, v599);
      pipe_barrier(PIPE_V);
      TADD(v598, v582, v596);
      // pto: %t__tmp_v149
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v600 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v149
      ;
      uint64_t v601 = (uint64_t) v36;
      TASSIGN(v600, v601);
      pipe_barrier(PIPE_V);
      TADDS(v600, v598, v91);
      // pto: %post_ids_inline351__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v602 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %post_ids_inline351__ssa_v0
      ;
      uint64_t v603 = (uint64_t) v36;
      TASSIGN(v602, v603);
      pipe_barrier(PIPE_V);
      RoundMode v604 = RoundMode::CAST_ROUND;
      SaturationMode v605 = SaturationMode::ON;
      TCVT(v602, v600, v604, v605);
      // pto: %post_tmp_inline474__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v606 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %post_tmp_inline474__ssa_v0
      ;
      uint64_t v607 = (uint64_t) v45;
      TASSIGN(v606, v607);
      // pto: %t__tmp_v150
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v608 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v83);
      // pto: %t__tmp_v150
      ;
      uint64_t v609 = (uint64_t) v56;
      TASSIGN(v608, v609);
      // pto: %post_values_inline592__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v610 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %post_values_inline592__ssa_v0
      ;
      uint64_t v611 = (uint64_t) v46;
      TASSIGN(v610, v611);
      pipe_barrier(PIPE_V);
      TGATHER(v610, v608, v602, v606);
      // pto: %t__tmp_v151
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v612 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v151
      ;
      uint64_t v613 = (uint64_t) v36;
      TASSIGN(v612, v613);
      pipe_barrier(PIPE_V);
      TMULS(v612, v610, v92);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      // pto: %129
      ;
      int64_t v614 = (int64_t) ((uint64_t) v472 * (uint64_t) v93);
      v612.SetValidShape(v65, v614);
      // pto: %130
      ;
      bool v615 = v444 == v75;
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      if (v615) {
        // pto: %132
        ;
        int64_t v616 = (int64_t) ((uint64_t) v470 * (uint64_t) v93);
        // pto: %133
        ;
        int64_t v617 = v616 < v75 ? v75 : v616;
        // pto: %post_flat_inline459__ssa_v0_pview
        ;
        const int64_t v618 = 0;
        // pto: %post_flat_inline459__ssa_v0_pview
        ;
        __gm__ float* v619 = PTOAS__GLOBAL_TENSOR_DATA(v165);
        // pto: %post_flat_inline459__ssa_v0_pview
        ;
        const int64_t v620 = 1;
        // pto: %post_flat_inline459__ssa_v0_pview
        ;
        const int64_t v621 = 1;
        // pto: %post_flat_inline459__ssa_v0_pview
        ;
        const int64_t v622 = 1;
        // pto: %post_flat_inline459__ssa_v0_pview
        ;
        int64_t v623 = v65 * v157;
        // pto: %post_flat_inline459__ssa_v0_pview
        ;
        int64_t v624 = v622 * v623;
        // pto: %post_flat_inline459__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, -1> v625 = pto::Shape<1, 1, 1, 1, -1>(v620, v621, v622, v65, v614);
        // pto: %post_flat_inline459__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v626 = pto::Stride<-1, -1, -1, -1, -1>(v621 * v624, v624, v623, v157, v65);
        // pto: %post_flat_inline459__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v627 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v619 + (v618 + v75 * v157 + v617 * v65), v625, v626);
        TSTORE(v627, v612);
      };
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      // pto: %134
      ;
      uint64_t v628 = (uint64_t) v444;
      // pto: %134
      ;
      int64_t v629 = (int64_t) (v628 * (uint64_t) v77);
      // pto: %135
      ;
      uint64_t v630 = (uint64_t) v445;
      // pto: %135
      ;
      uint64_t v631 = (uint64_t) v444;
      // pto: %135, %136
      ;
      int64_t v632 = (int64_t) ((uint64_t) ((int64_t) v630 - v631) * (uint64_t) v77);
      // pto: %137
      ;
      int64_t v633 = v472 / v76;
      // pto: %138
      ;
      int64_t v634 = (int64_t) ((uint64_t) v633 * (uint64_t) v76);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      for (int64_t v635 = v75; v635 < v634; v635 += v76) {
        // pto: %139
        ;
        int64_t v636 = (int64_t) ((uint64_t) v470 + (uint64_t) v635);
        // pto: %140
        ;
        int64_t v637 = (int64_t) ((uint64_t) v635 + (uint64_t) v65);
        // pto: %141
        ;
        int64_t v638 = (int64_t) ((uint64_t) v470 + (uint64_t) v637);
        // pto: %flat_offset_mul
        ;
        int64_t v639 = (int64_t) ((uint64_t) v635 * (uint64_t) v79);
        // pto: %t__tmp_v152
        ;
        float v640 = v574.GetValue(v639);
        // pto: %142
        ;
        float v641 = v640 + v81;
        // pto: %144, %t__tmp_v153
        ;
        float v642 = v574.GetValue((int64_t) ((uint64_t) v639 + (uint64_t) v65));
        // pto: %145
        ;
        float v643 = v642 + v81;
        // pto: %147, %t__tmp_v154
        ;
        float v644 = v574.GetValue((int64_t) ((uint64_t) v639 + (uint64_t) v76));
        // pto: %148
        ;
        float v645 = v644 + v81;
        // pto: %150, %t__tmp_v155
        ;
        float v646 = v574.GetValue((int64_t) ((uint64_t) v639 + (uint64_t) v68));
        // pto: %151
        ;
        float v647 = v646 + v81;
        // pto: %t__tmp_v156
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v648 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %t__tmp_v156
        ;
        uint64_t v649 = (uint64_t) v57;
        TASSIGN(v648, v649);
        // pto: %152
        ;
        int64_t v650 = v636 < v75 ? v75 : v636;
        // pto: %153
        ;
        int64_t v651 = v629 < v75 ? v75 : v629;
        // pto: %154
        ;
        const int64_t v652 = 0;
        // pto: %154
        ;
        __gm__ bfloat16_t* v653 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %154
        ;
        const int64_t v654 = 1;
        // pto: %154
        ;
        const int64_t v655 = 1;
        // pto: %154
        ;
        const int64_t v656 = 1;
        // pto: %154
        ;
        int64_t v657 = v65 * v67;
        // pto: %154
        ;
        int64_t v658 = v656 * v657;
        // pto: %154
        ;
        pto::Shape<1, 1, 1, 1, -1> v659 = pto::Shape<1, 1, 1, 1, -1>(v654, v655, v656, v65, v632);
        // pto: %154
        ;
        pto::Stride<-1, -1, -1, -1, -1> v660 = pto::Stride<-1, -1, -1, -1, -1>(v655 * v658, v658, v657, v67, v65);
        // pto: %154
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v661 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v653 + (v652 + v650 * v67 + v651 * v65), v659, v660);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
        TLOAD(v648, v661);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        // pto: %t__tmp_v157
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v662 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %t__tmp_v157
        ;
        uint64_t v663 = (uint64_t) v58;
        TASSIGN(v662, v663);
        // pto: %156
        ;
        int64_t v664 = (int64_t) ((uint64_t) v629 + (uint64_t) v70);
        // pto: %157
        ;
        int64_t v665 = v664 < v75 ? v75 : v664;
        // pto: %158
        ;
        const int64_t v666 = 0;
        // pto: %158
        ;
        __gm__ bfloat16_t* v667 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %158
        ;
        const int64_t v668 = 1;
        // pto: %158
        ;
        const int64_t v669 = 1;
        // pto: %158
        ;
        const int64_t v670 = 1;
        // pto: %158
        ;
        int64_t v671 = v65 * v67;
        // pto: %158
        ;
        int64_t v672 = v670 * v671;
        // pto: %158
        ;
        pto::Shape<1, 1, 1, 1, -1> v673 = pto::Shape<1, 1, 1, 1, -1>(v668, v669, v670, v65, v632);
        // pto: %158
        ;
        pto::Stride<-1, -1, -1, -1, -1> v674 = pto::Stride<-1, -1, -1, -1, -1>(v669 * v672, v672, v671, v67, v65);
        // pto: %158
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v675 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v667 + (v666 + v650 * v67 + v665 * v65), v673, v674);
        TLOAD(v662, v675);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %t__tmp_v158
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v676 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %t__tmp_v158
        ;
        uint64_t v677 = (uint64_t) v59;
        TASSIGN(v676, v677);
        // pto: %160
        ;
        int64_t v678 = (int64_t) ((uint64_t) v629 + (uint64_t) v94);
        // pto: %161
        ;
        int64_t v679 = v678 < v75 ? v75 : v678;
        // pto: %162
        ;
        const int64_t v680 = 0;
        // pto: %162
        ;
        __gm__ bfloat16_t* v681 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %162
        ;
        const int64_t v682 = 1;
        // pto: %162
        ;
        const int64_t v683 = 1;
        // pto: %162
        ;
        const int64_t v684 = 1;
        // pto: %162
        ;
        int64_t v685 = v65 * v67;
        // pto: %162
        ;
        int64_t v686 = v684 * v685;
        // pto: %162
        ;
        pto::Shape<1, 1, 1, 1, -1> v687 = pto::Shape<1, 1, 1, 1, -1>(v682, v683, v684, v65, v632);
        // pto: %162
        ;
        pto::Stride<-1, -1, -1, -1, -1> v688 = pto::Stride<-1, -1, -1, -1, -1>(v683 * v686, v686, v685, v67, v65);
        // pto: %162
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v689 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v681 + (v680 + v650 * v67 + v679 * v65), v687, v688);
        TLOAD(v676, v689);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %t__tmp_v159
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v690 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %t__tmp_v159
        ;
        uint64_t v691 = (uint64_t) v34;
        TASSIGN(v690, v691);
        // pto: %164
        ;
        int64_t v692 = (int64_t) ((uint64_t) v629 + (uint64_t) v95);
        // pto: %165
        ;
        int64_t v693 = v692 < v75 ? v75 : v692;
        // pto: %166
        ;
        const int64_t v694 = 0;
        // pto: %166
        ;
        __gm__ bfloat16_t* v695 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %166
        ;
        const int64_t v696 = 1;
        // pto: %166
        ;
        const int64_t v697 = 1;
        // pto: %166
        ;
        const int64_t v698 = 1;
        // pto: %166
        ;
        int64_t v699 = v65 * v67;
        // pto: %166
        ;
        int64_t v700 = v698 * v699;
        // pto: %166
        ;
        pto::Shape<1, 1, 1, 1, -1> v701 = pto::Shape<1, 1, 1, 1, -1>(v696, v697, v698, v65, v632);
        // pto: %166
        ;
        pto::Stride<-1, -1, -1, -1, -1> v702 = pto::Stride<-1, -1, -1, -1, -1>(v697 * v700, v700, v699, v67, v65);
        // pto: %166
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v703 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v695 + (v694 + v650 * v67 + v693 * v65), v701, v702);
        TLOAD(v690, v703);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %169
        ;
        int64_t v704 = (int64_t) ((uint64_t) v637 * (uint64_t) v79);
        // pto: %167
        ;
        float v705 = v574.GetValue(v704);
        // pto: %171
        ;
        float v706 = v705 + v81;
        // pto: %175, %172
        ;
        float v707 = v574.GetValue((int64_t) ((uint64_t) v704 + (uint64_t) v65));
        // pto: %176
        ;
        float v708 = v707 + v81;
        // pto: %180, %177
        ;
        float v709 = v574.GetValue((int64_t) ((uint64_t) v704 + (uint64_t) v76));
        // pto: %181
        ;
        float v710 = v709 + v81;
        // pto: %185, %182
        ;
        float v711 = v574.GetValue((int64_t) ((uint64_t) v704 + (uint64_t) v68));
        // pto: %186
        ;
        float v712 = v711 + v81;
        // pto: %4
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v713 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %4
        ;
        uint64_t v714 = (uint64_t) v60;
        TASSIGN(v713, v714);
        // pto: %187
        ;
        int64_t v715 = v638 < v75 ? v75 : v638;
        // pto: %189
        ;
        const int64_t v716 = 0;
        // pto: %189
        ;
        __gm__ bfloat16_t* v717 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %189
        ;
        const int64_t v718 = 1;
        // pto: %189
        ;
        const int64_t v719 = 1;
        // pto: %189
        ;
        const int64_t v720 = 1;
        // pto: %189
        ;
        int64_t v721 = v65 * v67;
        // pto: %189
        ;
        int64_t v722 = v720 * v721;
        // pto: %189
        ;
        pto::Shape<1, 1, 1, 1, -1> v723 = pto::Shape<1, 1, 1, 1, -1>(v718, v719, v720, v65, v632);
        // pto: %189
        ;
        pto::Stride<-1, -1, -1, -1, -1> v724 = pto::Stride<-1, -1, -1, -1, -1>(v719 * v722, v722, v721, v67, v65);
        // pto: %189
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v725 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v717 + (v716 + v715 * v67 + v651 * v65), v723, v724);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v713, v725);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %5
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v726 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %5
        ;
        uint64_t v727 = (uint64_t) v61;
        TASSIGN(v726, v727);
        // pto: %193
        ;
        const int64_t v728 = 0;
        // pto: %193
        ;
        __gm__ bfloat16_t* v729 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %193
        ;
        const int64_t v730 = 1;
        // pto: %193
        ;
        const int64_t v731 = 1;
        // pto: %193
        ;
        const int64_t v732 = 1;
        // pto: %193
        ;
        int64_t v733 = v65 * v67;
        // pto: %193
        ;
        int64_t v734 = v732 * v733;
        // pto: %193
        ;
        pto::Shape<1, 1, 1, 1, -1> v735 = pto::Shape<1, 1, 1, 1, -1>(v730, v731, v732, v65, v632);
        // pto: %193
        ;
        pto::Stride<-1, -1, -1, -1, -1> v736 = pto::Stride<-1, -1, -1, -1, -1>(v731 * v734, v734, v733, v67, v65);
        // pto: %193
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v737 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v729 + (v728 + v715 * v67 + v665 * v65), v735, v736);
        TLOAD(v726, v737);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        // pto: %6
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v738 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %6
        ;
        uint64_t v739 = (uint64_t) v62;
        TASSIGN(v738, v739);
        // pto: %197
        ;
        const int64_t v740 = 0;
        // pto: %197
        ;
        __gm__ bfloat16_t* v741 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %197
        ;
        const int64_t v742 = 1;
        // pto: %197
        ;
        const int64_t v743 = 1;
        // pto: %197
        ;
        const int64_t v744 = 1;
        // pto: %197
        ;
        int64_t v745 = v65 * v67;
        // pto: %197
        ;
        int64_t v746 = v744 * v745;
        // pto: %197
        ;
        pto::Shape<1, 1, 1, 1, -1> v747 = pto::Shape<1, 1, 1, 1, -1>(v742, v743, v744, v65, v632);
        // pto: %197
        ;
        pto::Stride<-1, -1, -1, -1, -1> v748 = pto::Stride<-1, -1, -1, -1, -1>(v743 * v746, v746, v745, v67, v65);
        // pto: %197
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v749 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v741 + (v740 + v715 * v67 + v679 * v65), v747, v748);
        TLOAD(v738, v749);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        // pto: %7
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v750 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %7
        ;
        uint64_t v751 = (uint64_t) v63;
        TASSIGN(v750, v751);
        // pto: %201
        ;
        const int64_t v752 = 0;
        // pto: %201
        ;
        __gm__ bfloat16_t* v753 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %201
        ;
        const int64_t v754 = 1;
        // pto: %201
        ;
        const int64_t v755 = 1;
        // pto: %201
        ;
        const int64_t v756 = 1;
        // pto: %201
        ;
        int64_t v757 = v65 * v67;
        // pto: %201
        ;
        int64_t v758 = v756 * v757;
        // pto: %201
        ;
        pto::Shape<1, 1, 1, 1, -1> v759 = pto::Shape<1, 1, 1, 1, -1>(v754, v755, v756, v65, v632);
        // pto: %201
        ;
        pto::Stride<-1, -1, -1, -1, -1> v760 = pto::Stride<-1, -1, -1, -1, -1>(v755 * v758, v758, v757, v67, v65);
        // pto: %201
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v761 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v753 + (v752 + v715 * v67 + v693 * v65), v759, v760);
        TLOAD(v750, v761);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        // pto: %part_scalar_x0_inline576__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v762 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %part_scalar_x0_inline576__ssa_v0
        ;
        uint64_t v763 = (uint64_t) v36;
        TASSIGN(v762, v763);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        RoundMode v764 = RoundMode::CAST_ROUND;
        SaturationMode v765 = SaturationMode::ON;
        TCVT(v762, v648, v764, v765);
        // pto: %part_scalar_y0_inline441__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v766 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %part_scalar_y0_inline441__ssa_v0
        ;
        uint64_t v767 = (uint64_t) v36;
        TASSIGN(v766, v767);
        pipe_barrier(PIPE_V);
        TMULS(v766, v762, v641);
        // pto: %part_scalar_x1_inline345__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v768 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %part_scalar_x1_inline345__ssa_v0
        ;
        uint64_t v769 = (uint64_t) v57;
        TASSIGN(v768, v769);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v770 = RoundMode::CAST_ROUND;
        SaturationMode v771 = SaturationMode::ON;
        TCVT(v768, v662, v770, v771);
        // pto: %part_scalar_y1_inline344__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v772 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %part_scalar_y1_inline344__ssa_v0
        ;
        uint64_t v773 = (uint64_t) v57;
        TASSIGN(v772, v773);
        pipe_barrier(PIPE_V);
        TMULS(v772, v768, v643);
        // pto: %part_scalar_sum01_inline343__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v774 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %part_scalar_sum01_inline343__ssa_v0
        ;
        uint64_t v775 = (uint64_t) v36;
        TASSIGN(v774, v775);
        pipe_barrier(PIPE_V);
        TADD(v774, v766, v772);
        // pto: %part_scalar_x2_inline342__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v776 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %part_scalar_x2_inline342__ssa_v0
        ;
        uint64_t v777 = (uint64_t) v57;
        TASSIGN(v776, v777);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        RoundMode v778 = RoundMode::CAST_ROUND;
        SaturationMode v779 = SaturationMode::ON;
        TCVT(v776, v676, v778, v779);
        // pto: %part_scalar_y2_inline341__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v780 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %part_scalar_y2_inline341__ssa_v0
        ;
        uint64_t v781 = (uint64_t) v57;
        TASSIGN(v780, v781);
        pipe_barrier(PIPE_V);
        TMULS(v780, v776, v645);
        // pto: %part_scalar_x3_inline438__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v782 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %part_scalar_x3_inline438__ssa_v0
        ;
        uint64_t v783 = (uint64_t) v58;
        TASSIGN(v782, v783);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        RoundMode v784 = RoundMode::CAST_ROUND;
        SaturationMode v785 = SaturationMode::ON;
        TCVT(v782, v690, v784, v785);
        // pto: %part_scalar_y3_inline340__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v786 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %part_scalar_y3_inline340__ssa_v0
        ;
        uint64_t v787 = (uint64_t) v58;
        TASSIGN(v786, v787);
        pipe_barrier(PIPE_V);
        TMULS(v786, v782, v647);
        // pto: %t__tmp_v160
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v788 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %t__tmp_v160
        ;
        uint64_t v789 = (uint64_t) v57;
        TASSIGN(v788, v789);
        pipe_barrier(PIPE_V);
        TADD(v788, v780, v786);
        // pto: %part_scalar_result_inline600__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v790 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %part_scalar_result_inline600__ssa_v0
        ;
        uint64_t v791 = (uint64_t) v36;
        TASSIGN(v790, v791);
        pipe_barrier(PIPE_V);
        TADD(v790, v774, v788);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
        // pto: %part_scalar_bf16_inline339__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v792 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %part_scalar_bf16_inline339__ssa_v0
        ;
        uint64_t v793 = (uint64_t) v36;
        TASSIGN(v792, v793);
        pipe_barrier(PIPE_V);
        RoundMode v794 = RoundMode::CAST_RINT;
        SaturationMode v795 = SaturationMode::ON;
        TCVT(v792, v790, v794, v795);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        v792.SetValidShape(v65, v632);
        // pto: %mixed_view_inline481__ssa_v0_pview
        ;
        const int64_t v796 = 0;
        // pto: %mixed_view_inline481__ssa_v0_pview
        ;
        __gm__ bfloat16_t* v797 = PTOAS__GLOBAL_TENSOR_DATA(v174);
        // pto: %mixed_view_inline481__ssa_v0_pview
        ;
        const int64_t v798 = 1;
        // pto: %mixed_view_inline481__ssa_v0_pview
        ;
        const int64_t v799 = 1;
        // pto: %mixed_view_inline481__ssa_v0_pview
        ;
        const int64_t v800 = 1;
        // pto: %mixed_view_inline481__ssa_v0_pview
        ;
        int64_t v801 = v65 * v70;
        // pto: %mixed_view_inline481__ssa_v0_pview
        ;
        int64_t v802 = v800 * v801;
        // pto: %mixed_view_inline481__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, -1> v803 = pto::Shape<1, 1, 1, 1, -1>(v798, v799, v800, v65, v632);
        // pto: %mixed_view_inline481__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v804 = pto::Stride<-1, -1, -1, -1, -1>(v799 * v802, v802, v801, v70, v65);
        // pto: %mixed_view_inline481__ssa_v0_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v805 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v797 + (v796 + v650 * v70 + v651 * v65), v803, v804);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v805, v792);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        // pto: %8
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v806 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %8
        ;
        uint64_t v807 = (uint64_t) v35;
        TASSIGN(v806, v807);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        RoundMode v808 = RoundMode::CAST_ROUND;
        SaturationMode v809 = SaturationMode::ON;
        TCVT(v806, v713, v808, v809);
        // pto: %9
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v810 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %9
        ;
        uint64_t v811 = (uint64_t) v35;
        TASSIGN(v810, v811);
        pipe_barrier(PIPE_V);
        TMULS(v810, v806, v706);
        // pto: %10
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v812 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %10
        ;
        uint64_t v813 = (uint64_t) v60;
        TASSIGN(v812, v813);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        RoundMode v814 = RoundMode::CAST_ROUND;
        SaturationMode v815 = SaturationMode::ON;
        TCVT(v812, v726, v814, v815);
        // pto: %11
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v816 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %11
        ;
        uint64_t v817 = (uint64_t) v60;
        TASSIGN(v816, v817);
        pipe_barrier(PIPE_V);
        TMULS(v816, v812, v708);
        // pto: %12
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v818 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %12
        ;
        uint64_t v819 = (uint64_t) v35;
        TASSIGN(v818, v819);
        pipe_barrier(PIPE_V);
        TADD(v818, v810, v816);
        // pto: %13
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v820 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %13
        ;
        uint64_t v821 = (uint64_t) v60;
        TASSIGN(v820, v821);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        RoundMode v822 = RoundMode::CAST_ROUND;
        SaturationMode v823 = SaturationMode::ON;
        TCVT(v820, v738, v822, v823);
        // pto: %14
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v824 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %14
        ;
        uint64_t v825 = (uint64_t) v60;
        TASSIGN(v824, v825);
        pipe_barrier(PIPE_V);
        TMULS(v824, v820, v710);
        // pto: %15
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v826 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %15
        ;
        uint64_t v827 = (uint64_t) v61;
        TASSIGN(v826, v827);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        RoundMode v828 = RoundMode::CAST_ROUND;
        SaturationMode v829 = SaturationMode::ON;
        TCVT(v826, v750, v828, v829);
        // pto: %16
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v830 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %16
        ;
        uint64_t v831 = (uint64_t) v61;
        TASSIGN(v830, v831);
        pipe_barrier(PIPE_V);
        TMULS(v830, v826, v712);
        // pto: %17
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v832 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %17
        ;
        uint64_t v833 = (uint64_t) v60;
        TASSIGN(v832, v833);
        pipe_barrier(PIPE_V);
        TADD(v832, v824, v830);
        // pto: %18
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v834 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %18
        ;
        uint64_t v835 = (uint64_t) v35;
        TASSIGN(v834, v835);
        pipe_barrier(PIPE_V);
        TADD(v834, v818, v832);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %19
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v836 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %19
        ;
        uint64_t v837 = (uint64_t) v35;
        TASSIGN(v836, v837);
        pipe_barrier(PIPE_V);
        RoundMode v838 = RoundMode::CAST_RINT;
        SaturationMode v839 = SaturationMode::ON;
        TCVT(v836, v834, v838, v839);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        v836.SetValidShape(v65, v632);
        // pto: %207
        ;
        const int64_t v840 = 0;
        // pto: %207
        ;
        __gm__ bfloat16_t* v841 = PTOAS__GLOBAL_TENSOR_DATA(v174);
        // pto: %207
        ;
        const int64_t v842 = 1;
        // pto: %207
        ;
        const int64_t v843 = 1;
        // pto: %207
        ;
        const int64_t v844 = 1;
        // pto: %207
        ;
        int64_t v845 = v65 * v70;
        // pto: %207
        ;
        int64_t v846 = v844 * v845;
        // pto: %207
        ;
        pto::Shape<1, 1, 1, 1, -1> v847 = pto::Shape<1, 1, 1, 1, -1>(v842, v843, v844, v65, v632);
        // pto: %207
        ;
        pto::Stride<-1, -1, -1, -1, -1> v848 = pto::Stride<-1, -1, -1, -1, -1>(v843 * v846, v846, v845, v70, v65);
        // pto: %207
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v849 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v841 + (v840 + v715 * v70 + v651 * v65), v847, v848);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v849, v836);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
      };
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_S, PIPE_V, EVENT_ID1);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
      // pto: %208, %209
      ;
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
      if ((int64_t) ((uint64_t) v472 - (uint64_t) v634) == v65) {
        // pto: %210
        ;
        int64_t v850 = (int64_t) ((uint64_t) v470 + (uint64_t) v634);
        int64_t v851 = (int64_t) ((uint64_t) v633 * (uint64_t) v69);
        // pto: %211
        ;
        float v852 = v574.GetValue(v851);
        // pto: %214
        ;
        float v853 = v852 + v81;
        // pto: %217, %215
        ;
        float v854 = v574.GetValue((int64_t) ((uint64_t) v851 + (uint64_t) v65));
        // pto: %218
        ;
        float v855 = v854 + v81;
        // pto: %221, %219
        ;
        float v856 = v574.GetValue((int64_t) ((uint64_t) v851 + (uint64_t) v76));
        // pto: %222
        ;
        float v857 = v856 + v81;
        // pto: %225, %223
        ;
        float v858 = v574.GetValue((int64_t) ((uint64_t) v851 + (uint64_t) v68));
        // pto: %226
        ;
        float v859 = v858 + v81;
        // pto: %21
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v860 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %21
        ;
        uint64_t v861 = (uint64_t) v57;
        TASSIGN(v860, v861);
        // pto: %227
        ;
        int64_t v862 = v850 < v75 ? v75 : v850;
        // pto: %228
        ;
        int64_t v863 = v629 < v75 ? v75 : v629;
        // pto: %229
        ;
        const int64_t v864 = 0;
        // pto: %229
        ;
        __gm__ bfloat16_t* v865 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %229
        ;
        const int64_t v866 = 1;
        // pto: %229
        ;
        const int64_t v867 = 1;
        // pto: %229
        ;
        const int64_t v868 = 1;
        // pto: %229
        ;
        int64_t v869 = v65 * v67;
        // pto: %229
        ;
        int64_t v870 = v868 * v869;
        // pto: %229
        ;
        pto::Shape<1, 1, 1, 1, -1> v871 = pto::Shape<1, 1, 1, 1, -1>(v866, v867, v868, v65, v632);
        // pto: %229
        ;
        pto::Stride<-1, -1, -1, -1, -1> v872 = pto::Stride<-1, -1, -1, -1, -1>(v867 * v870, v870, v869, v67, v65);
        // pto: %229
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v873 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v865 + (v864 + v862 * v67 + v863 * v65), v871, v872);
        TLOAD(v860, v873);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %22
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v874 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %22
        ;
        uint64_t v875 = (uint64_t) v36;
        TASSIGN(v874, v875);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v876 = RoundMode::CAST_ROUND;
        SaturationMode v877 = SaturationMode::ON;
        TCVT(v874, v860, v876, v877);
        // pto: %23
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v878 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %23
        ;
        uint64_t v879 = (uint64_t) v36;
        TASSIGN(v878, v879);
        pipe_barrier(PIPE_V);
        TMULS(v878, v874, v853);
        // pto: %24
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v880 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %24
        ;
        uint64_t v881 = (uint64_t) v58;
        TASSIGN(v880, v881);
        // pto: %231
        ;
        int64_t v882 = (int64_t) ((uint64_t) v629 + (uint64_t) v70);
        // pto: %232
        ;
        int64_t v883 = v882 < v75 ? v75 : v882;
        // pto: %233
        ;
        const int64_t v884 = 0;
        // pto: %233
        ;
        __gm__ bfloat16_t* v885 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %233
        ;
        const int64_t v886 = 1;
        // pto: %233
        ;
        const int64_t v887 = 1;
        // pto: %233
        ;
        const int64_t v888 = 1;
        // pto: %233
        ;
        int64_t v889 = v65 * v67;
        // pto: %233
        ;
        int64_t v890 = v888 * v889;
        // pto: %233
        ;
        pto::Shape<1, 1, 1, 1, -1> v891 = pto::Shape<1, 1, 1, 1, -1>(v886, v887, v888, v65, v632);
        // pto: %233
        ;
        pto::Stride<-1, -1, -1, -1, -1> v892 = pto::Stride<-1, -1, -1, -1, -1>(v887 * v890, v890, v889, v67, v65);
        // pto: %233
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v893 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v885 + (v884 + v862 * v67 + v883 * v65), v891, v892);
        TLOAD(v880, v893);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %25
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v894 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %25
        ;
        uint64_t v895 = (uint64_t) v57;
        TASSIGN(v894, v895);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v896 = RoundMode::CAST_ROUND;
        SaturationMode v897 = SaturationMode::ON;
        TCVT(v894, v880, v896, v897);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %26
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v898 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %26
        ;
        uint64_t v899 = (uint64_t) v57;
        TASSIGN(v898, v899);
        pipe_barrier(PIPE_V);
        TMULS(v898, v894, v855);
        // pto: %27
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v900 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %27
        ;
        uint64_t v901 = (uint64_t) v36;
        TASSIGN(v900, v901);
        pipe_barrier(PIPE_V);
        TADD(v900, v878, v898);
        // pto: %28
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v902 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %28
        ;
        uint64_t v903 = (uint64_t) v58;
        TASSIGN(v902, v903);
        // pto: %235
        ;
        int64_t v904 = (int64_t) ((uint64_t) v629 + (uint64_t) v94);
        // pto: %236
        ;
        int64_t v905 = v904 < v75 ? v75 : v904;
        // pto: %237
        ;
        const int64_t v906 = 0;
        // pto: %237
        ;
        __gm__ bfloat16_t* v907 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %237
        ;
        const int64_t v908 = 1;
        // pto: %237
        ;
        const int64_t v909 = 1;
        // pto: %237
        ;
        const int64_t v910 = 1;
        // pto: %237
        ;
        int64_t v911 = v65 * v67;
        // pto: %237
        ;
        int64_t v912 = v910 * v911;
        // pto: %237
        ;
        pto::Shape<1, 1, 1, 1, -1> v913 = pto::Shape<1, 1, 1, 1, -1>(v908, v909, v910, v65, v632);
        // pto: %237
        ;
        pto::Stride<-1, -1, -1, -1, -1> v914 = pto::Stride<-1, -1, -1, -1, -1>(v909 * v912, v912, v911, v67, v65);
        // pto: %237
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v915 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v907 + (v906 + v862 * v67 + v905 * v65), v913, v914);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v902, v915);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %29
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v916 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %29
        ;
        uint64_t v917 = (uint64_t) v57;
        TASSIGN(v916, v917);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v918 = RoundMode::CAST_ROUND;
        SaturationMode v919 = SaturationMode::ON;
        TCVT(v916, v902, v918, v919);
        // pto: %30
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v920 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %30
        ;
        uint64_t v921 = (uint64_t) v57;
        TASSIGN(v920, v921);
        pipe_barrier(PIPE_V);
        TMULS(v920, v916, v857);
        // pto: %31
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v922 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %31
        ;
        uint64_t v923 = (uint64_t) v35;
        TASSIGN(v922, v923);
        // pto: %239
        ;
        int64_t v924 = (int64_t) ((uint64_t) v629 + (uint64_t) v95);
        // pto: %240
        ;
        int64_t v925 = v924 < v75 ? v75 : v924;
        // pto: %241
        ;
        const int64_t v926 = 0;
        // pto: %241
        ;
        __gm__ bfloat16_t* v927 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %241
        ;
        const int64_t v928 = 1;
        // pto: %241
        ;
        const int64_t v929 = 1;
        // pto: %241
        ;
        const int64_t v930 = 1;
        // pto: %241
        ;
        int64_t v931 = v65 * v67;
        // pto: %241
        ;
        int64_t v932 = v930 * v931;
        // pto: %241
        ;
        pto::Shape<1, 1, 1, 1, -1> v933 = pto::Shape<1, 1, 1, 1, -1>(v928, v929, v930, v65, v632);
        // pto: %241
        ;
        pto::Stride<-1, -1, -1, -1, -1> v934 = pto::Stride<-1, -1, -1, -1, -1>(v929 * v932, v932, v931, v67, v65);
        // pto: %241
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v935 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v927 + (v926 + v862 * v67 + v925 * v65), v933, v934);
        TLOAD(v922, v935);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %32
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v936 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %32
        ;
        uint64_t v937 = (uint64_t) v58;
        TASSIGN(v936, v937);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v938 = RoundMode::CAST_ROUND;
        SaturationMode v939 = SaturationMode::ON;
        TCVT(v936, v922, v938, v939);
        // pto: %33
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v940 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %33
        ;
        uint64_t v941 = (uint64_t) v58;
        TASSIGN(v940, v941);
        pipe_barrier(PIPE_V);
        TMULS(v940, v936, v859);
        // pto: %34
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v942 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %34
        ;
        uint64_t v943 = (uint64_t) v57;
        TASSIGN(v942, v943);
        pipe_barrier(PIPE_V);
        TADD(v942, v920, v940);
        // pto: %35
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v944 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %35
        ;
        uint64_t v945 = (uint64_t) v36;
        TASSIGN(v944, v945);
        pipe_barrier(PIPE_V);
        TADD(v944, v900, v942);
        // pto: %36
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v946 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v632);
        // pto: %36
        ;
        uint64_t v947 = (uint64_t) v36;
        TASSIGN(v946, v947);
        pipe_barrier(PIPE_V);
        RoundMode v948 = RoundMode::CAST_RINT;
        SaturationMode v949 = SaturationMode::ON;
        TCVT(v946, v944, v948, v949);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        v946.SetValidShape(v65, v632);
        // pto: %245
        ;
        const int64_t v950 = 0;
        // pto: %245
        ;
        __gm__ bfloat16_t* v951 = PTOAS__GLOBAL_TENSOR_DATA(v174);
        // pto: %245
        ;
        const int64_t v952 = 1;
        // pto: %245
        ;
        const int64_t v953 = 1;
        // pto: %245
        ;
        const int64_t v954 = 1;
        // pto: %245
        ;
        int64_t v955 = v65 * v70;
        // pto: %245
        ;
        int64_t v956 = v954 * v955;
        // pto: %245
        ;
        pto::Shape<1, 1, 1, 1, -1> v957 = pto::Shape<1, 1, 1, 1, -1>(v952, v953, v954, v65, v632);
        // pto: %245
        ;
        pto::Stride<-1, -1, -1, -1, -1> v958 = pto::Stride<-1, -1, -1, -1, -1>(v953 * v956, v956, v955, v70, v65);
        // pto: %245
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v959 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v951 + (v950 + v862 * v70 + v863 * v65), v957, v958);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v959, v946);
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