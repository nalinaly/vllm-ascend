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

AICORE void hc_large144_aic(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ bfloat16_t* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, __gm__ float* v10, __gm__ float* v11, __gm__ bfloat16_t* v12, __gm__ float* v13, int64_t v14, int32_t v15, int64_t v16, int64_t v17, int64_t v18, int64_t v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int64_t v28, int64_t v29, int64_t v30, int32_t v31, int32_t v32) {
  using T = float;

  #if defined(__DAV_CUBE__)
  const int64_t v33 = 384;
  const int32_t v34 = 0;
  // pto: %c0_i64
  const int64_t v35 = 0;
  // pto: %c196608_i64
  const int64_t v36 = 196608;
  // pto: %c262144_i64
  const int64_t v37 = 262144;
  // pto: %c24576_i64
  const int64_t v38 = 24576;
  // pto: %c16384_i64
  const int64_t v39 = 16384;
  // pto: %c32768_i64
  const int64_t v40 = 32768;
  // pto: %c49152_i64
  const int64_t v41 = 49152;
  // pto: %c32_index
  const int64_t v42 = 32;
  // pto: %c1_index
  const int64_t v43 = 1;
  // pto: %c24_index
  const int64_t v44 = 24;
  // pto: %c16384_index
  const int64_t v45 = 16384;
  // pto: %c8_index
  const int64_t v46 = 8;
  // pto: %c48_index
  const int64_t v47 = 48;
  // pto: %c2048_index
  const int64_t v48 = 2048;
  // pto: %c0_index
  const int64_t v49 = 0;
  // pto: %c4_index
  const int64_t v50 = 4;
  // pto: %c2_index
  const int64_t v51 = 2;
  // pto: %c512_index
  const int64_t v52 = 512;
  // pto: %c256_index
  const int64_t v53 = 256;
  // pto: %c128_index
  const int64_t v54 = 128;
  // pto: %cn128_index
  const int64_t v55 = -128;
  // pto: %63
  int64_t v56 = (int64_t) ((uint64_t) v29 * (uint64_t) v46);
  // pto: %mixes_partials_inline1438__ssa_v0_view
  const int64_t v57 = 1;
  // pto: %mixes_partials_inline1438__ssa_v0_view
  const int64_t v58 = 1;
  // pto: %mixes_partials_inline1438__ssa_v0_view
  const int64_t v59 = 1;
  // pto: %mixes_partials_inline1438__ssa_v0_view
  int64_t v60 = v56 * v42;
  // pto: %mixes_partials_inline1438__ssa_v0_view
  int64_t v61 = v59 * v60;
  // pto: %mixes_partials_inline1438__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v62 = pto::Shape<1, 1, 1, -1, -1>(v57, v58, v59, v56, v42);
  // pto: %mixes_partials_inline1438__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v63 = pto::Stride<-1, -1, -1, -1, -1>(v58 * v61, v61, v60, v42, v43);
  // pto: %mixes_partials_inline1438__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v64 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v62, v63);
  // pto: %hc_fn__ssa_v0_view
  const int64_t v65 = 1;
  // pto: %hc_fn__ssa_v0_view
  const int64_t v66 = 1;
  // pto: %hc_fn__ssa_v0_view
  const int64_t v67 = 1;
  // pto: %hc_fn__ssa_v0_view
  int64_t v68 = v44 * v45;
  // pto: %hc_fn__ssa_v0_view
  int64_t v69 = v67 * v68;
  // pto: %hc_fn__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v70 = pto::Shape<1, 1, 1, -1, -1>(v65, v66, v67, v44, v45);
  // pto: %hc_fn__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v71 = pto::Stride<-1, -1, -1, -1, -1>(v66 * v69, v69, v68, v45, v43);
  // pto: %hc_fn__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v72 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v70, v71);
  // pto: %mid_stream_product_inline1372__phi_v5
  Tile<TileType::Acc, float, 48, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v73 = Tile<TileType::Acc, float, 48, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v47, v44);
  // pto: %mid_stream_product_inline1372__phi_v5
  uint64_t v74 = (uint64_t) v35;
  TASSIGN(v73, v74);
  const int32_t v75 = 0;
  auto v76 = TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>(v13, v75, v34);
  // pto: %worker_inline1431__ssa_v0
  // pto: %linear_workers_inline1457__ssa_v0_idx
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
  for (int64_t v77 = (int64_t) v31; v77 < v14; v77 += (int64_t) v15) {
    // pto: %20
    ;
    int64_t v78 = v77 % v46;
    // pto: %21
    ;
    int64_t v79 = (int64_t) ((uint64_t) v78 * (uint64_t) v48);
    // pto: %22, %23, %18, %19
    ;
    int64_t v80 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v78 * (uint64_t) v16) + (uint64_t) ((int64_t) (uint64_t) (v77 / v46) * (uint64_t) v47));
    // pto: %mid_stream_product_inline1372__tile
    ;
    Tile<TileType::Acc, float, 48, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v81 = Tile<TileType::Acc, float, 48, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v47, v42);
    // pto: %mid_stream_product_inline1372__tile
    ;
    uint64_t v82 = (uint64_t) v35;
    TASSIGN(v81, v82);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int64_t v83 = v49; v83 < v50; v83 += v51) {
      // pto: %24
      ;
      int64_t v84 = (int64_t) ((uint64_t) v83 * (uint64_t) v52);
      // pto: %25
      ;
      int64_t v85 = (int64_t) ((uint64_t) v79 + (uint64_t) v84);
      // pto: %28, %27
      ;
      int64_t v86 = (int64_t) ((uint64_t) v79 + (uint64_t) ((int64_t) (uint64_t) v84 + (uint64_t) v52));
      // pto: %mid_weight_chunk_inline1375__tile
      ;
      Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v87 = Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v44, v52);
      // pto: %mid_weight_chunk_inline1375__tile
      ;
      uint64_t v88 = (uint64_t) v36;
      TASSIGN(v87, v88);
      // pto: %29
      ;
      int64_t v89 = v85 < v49 ? v49 : v85;
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v90 = 0;
      // pto: %hc_fn__ssa_v0_pview
      ;
      __gm__ float* v91 = PTOAS__GLOBAL_TENSOR_DATA(v72);
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v92 = 1;
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v93 = 1;
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v94 = 1;
      // pto: %hc_fn__ssa_v0_pview
      ;
      int64_t v95 = v44 * v45;
      // pto: %hc_fn__ssa_v0_pview
      ;
      int64_t v96 = v94 * v95;
      // pto: %hc_fn__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 24, -1> v97 = pto::Shape<1, 1, 1, 24, -1>(v92, v93, v94, v44, v52);
      // pto: %hc_fn__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v98 = pto::Stride<-1, -1, -1, -1, -1>(v93 * v96, v96, v95, v45, v43);
      // pto: %hc_fn__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v99 = GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v91 + (v90 + v49 * v45 + v89 * v43), v97, v98);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v87, v99);
      // pto: %0
      ;
      Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v100 = Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v44, v52);
      // pto: %0
      ;
      uint64_t v101 = (uint64_t) v37;
      TASSIGN(v100, v101);
      // pto: %30
      ;
      int64_t v102 = v86 < v49 ? v49 : v86;
      // pto: %31
      ;
      const int64_t v103 = 0;
      // pto: %31
      ;
      __gm__ float* v104 = PTOAS__GLOBAL_TENSOR_DATA(v72);
      // pto: %31
      ;
      const int64_t v105 = 1;
      // pto: %31
      ;
      const int64_t v106 = 1;
      // pto: %31
      ;
      const int64_t v107 = 1;
      // pto: %31
      ;
      int64_t v108 = v44 * v45;
      // pto: %31
      ;
      int64_t v109 = v107 * v108;
      // pto: %31
      ;
      pto::Shape<1, 1, 1, 24, -1> v110 = pto::Shape<1, 1, 1, 24, -1>(v105, v106, v107, v44, v52);
      // pto: %31
      ;
      pto::Stride<-1, -1, -1, -1, -1> v111 = pto::Stride<-1, -1, -1, -1, -1>(v106 * v109, v109, v108, v45, v43);
      // pto: %31
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v112 = GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v104 + (v103 + v49 * v45 + v102 * v43), v110, v111);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      TLOAD(v100, v112);
      // pto: %mid_gathered_chunk_inline1537__tile
      ;
      const int32_t v113 = 48;
      // pto: %mid_gathered_chunk_inline1537__tile
      ;
      const int32_t v114 = 512;
      // pto: %mid_gathered_chunk_inline1537__tile
      ;
      Tile<TileType::Mat, float, 48, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v115 = Tile<TileType::Mat, float, 48, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v113, v114);
      pipe_barrier(PIPE_MTE2);
      TPOP<TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>, Tile<TileType::Mat, float, 48, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v76, v115);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %32
      ;
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      if (v83 == v49) {
        // pto: %mid_weight_chunk_inline1375__tile_t
        ;
        Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v116 = Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v52, v44);
        // pto: %mid_weight_chunk_inline1375__tile_t
        ;
        uint64_t v117 = (uint64_t) v36;
        TASSIGN(v116, v117);
        // pto: %mid_stream_product_inline1372__tile_l0_init_storage
        ;
        Tile<TileType::Acc, float, 48, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v118 = Tile<TileType::Acc, float, 48, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v47, v42);
        // pto: %mid_stream_product_inline1372__tile_l0_init_storage
        ;
        uint64_t v119 = (uint64_t) v35;
        TASSIGN(v118, v119);
        v118.SetValidShape(v47, v44);
        for (int64_t v120 = v49; v120 < v52; v120 += v53) {
          // pto: %mid_stream_product_inline1372__tile_l0_a
          ;
          Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v121 = Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v47, v54);
          // pto: %mid_stream_product_inline1372__tile_l0_a
          ;
          uint64_t v122 = (uint64_t) v35;
          TASSIGN(v121, v122);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          pipe_barrier(PIPE_MTE1);
          uint16_t v123 = static_cast<uint16_t>(v49);
          uint16_t v124 = static_cast<uint16_t>(v120);
          TEXTRACT(v121, v115, v123, v124);
          // pto: %33
          ;
          int64_t v125 = (int64_t) ((uint64_t) v52 - (uint64_t) v120);
          // pto: %34
          ;
          int64_t v126 = v125 < v54 ? v125 : v54;
          // pto: %mid_stream_product_inline1372__tile_l0_b
          ;
          Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v127 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v126, v44);
          // pto: %mid_stream_product_inline1372__tile_l0_b
          ;
          uint64_t v128 = (uint64_t) v35;
          TASSIGN(v127, v128);
          uint16_t v129 = static_cast<uint16_t>(v120);
          uint16_t v130 = static_cast<uint16_t>(v49);
          TEXTRACT(v127, v116, v129, v130);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          // pto: %1
          ;
          Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v131 = Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v47, v54);
          // pto: %1
          ;
          uint64_t v132 = (uint64_t) v38;
          TASSIGN(v131, v132);
          // pto: %35
          ;
          int64_t v133 = (int64_t) ((uint64_t) v120 + (uint64_t) v54);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
          uint16_t v134 = static_cast<uint16_t>(v49);
          uint16_t v135 = static_cast<uint16_t>(v133);
          TEXTRACT(v131, v115, v134, v135);
          int64_t v136 = (int64_t) ((uint64_t) v33 - (uint64_t) v120);
          // pto: %38
          ;
          int64_t v137 = v136 < v49 ? v49 : v136;
          // pto: %39
          ;
          int64_t v138 = v137 < v54 ? v137 : v54;
          // pto: %2
          ;
          Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v139 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v138, v44);
          // pto: %2
          ;
          uint64_t v140 = (uint64_t) v39;
          TASSIGN(v139, v140);
          uint16_t v141 = static_cast<uint16_t>(v133);
          uint16_t v142 = static_cast<uint16_t>(v49);
          TEXTRACT(v139, v116, v141, v142);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          // pto: %41
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          if (v120 == v49) {
            pipe_barrier(PIPE_M);
            TMATMUL(v118, v121, v127);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v118, v118, v121, v127);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          // pto: %42
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          if (v120 == v55) {
            pipe_barrier(PIPE_M);
            TMATMUL(v118, v131, v139);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v118, v118, v131, v139);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        };
      } else {
        // pto: %4
        ;
        Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v143 = Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v52, v44);
        // pto: %4
        ;
        uint64_t v144 = (uint64_t) v36;
        TASSIGN(v143, v144);
        for (int64_t v145 = v49; v145 < v52; v145 += v53) {
          // pto: %5
          ;
          Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v146 = Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v47, v54);
          // pto: %5
          ;
          uint64_t v147 = (uint64_t) v35;
          TASSIGN(v146, v147);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
          uint16_t v148 = static_cast<uint16_t>(v49);
          uint16_t v149 = static_cast<uint16_t>(v145);
          TEXTRACT(v146, v115, v148, v149);
          // pto: %44
          ;
          int64_t v150 = (int64_t) ((uint64_t) v52 - (uint64_t) v145);
          // pto: %45
          ;
          int64_t v151 = v150 < v54 ? v150 : v54;
          // pto: %6
          ;
          Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v152 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v151, v44);
          // pto: %6
          ;
          uint64_t v153 = (uint64_t) v35;
          TASSIGN(v152, v153);
          uint16_t v154 = static_cast<uint16_t>(v145);
          uint16_t v155 = static_cast<uint16_t>(v49);
          TEXTRACT(v152, v143, v154, v155);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
          // pto: %7
          ;
          Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v156 = Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v47, v54);
          // pto: %7
          ;
          uint64_t v157 = (uint64_t) v38;
          TASSIGN(v156, v157);
          // pto: %46
          ;
          int64_t v158 = (int64_t) ((uint64_t) v145 + (uint64_t) v54);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
          uint16_t v159 = static_cast<uint16_t>(v49);
          uint16_t v160 = static_cast<uint16_t>(v158);
          TEXTRACT(v156, v115, v159, v160);
          int64_t v161 = (int64_t) ((uint64_t) v33 - (uint64_t) v145);
          // pto: %49
          ;
          int64_t v162 = v161 < v49 ? v49 : v161;
          // pto: %50
          ;
          int64_t v163 = v162 < v54 ? v162 : v54;
          // pto: %8
          ;
          Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v164 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v163, v44);
          // pto: %8
          ;
          uint64_t v165 = (uint64_t) v39;
          TASSIGN(v164, v165);
          uint16_t v166 = static_cast<uint16_t>(v158);
          uint16_t v167 = static_cast<uint16_t>(v49);
          TEXTRACT(v164, v143, v166, v167);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
          pipe_barrier(PIPE_M);
          TMATMUL_ACC(v81, v81, v146, v152);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
          pipe_barrier(PIPE_M);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
          TMATMUL_ACC(v81, v81, v156, v164);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        };
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TFREE<TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>, TileSplitAxis::TILE_UP_DOWN>(v76);
      // pto: %52
      ;
      const int32_t v168 = 48;
      // pto: %52
      ;
      const int32_t v169 = 512;
      // pto: %52
      ;
      Tile<TileType::Mat, float, 48, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v170 = Tile<TileType::Mat, float, 48, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v168, v169);
      TPOP<TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>, Tile<TileType::Mat, float, 48, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v76, v170);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      // pto: %11
      ;
      Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v171 = Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v52, v44);
      // pto: %11
      ;
      uint64_t v172 = (uint64_t) v37;
      TASSIGN(v171, v172);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
      for (int64_t v173 = v49; v173 < v52; v173 += v53) {
        // pto: %12
        ;
        Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v174 = Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v47, v54);
        // pto: %12
        ;
        uint64_t v175 = (uint64_t) v35;
        TASSIGN(v174, v175);
        pipe_barrier(PIPE_MTE1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
        uint16_t v176 = static_cast<uint16_t>(v49);
        uint16_t v177 = static_cast<uint16_t>(v173);
        TEXTRACT(v174, v170, v176, v177);
        // pto: %54
        ;
        int64_t v178 = (int64_t) ((uint64_t) v52 - (uint64_t) v173);
        // pto: %55
        ;
        int64_t v179 = v178 < v54 ? v178 : v54;
        // pto: %13
        ;
        Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v180 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v179, v44);
        // pto: %13
        ;
        uint64_t v181 = (uint64_t) v40;
        TASSIGN(v180, v181);
        uint16_t v182 = static_cast<uint16_t>(v173);
        uint16_t v183 = static_cast<uint16_t>(v49);
        TEXTRACT(v180, v171, v182, v183);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
        // pto: %14
        ;
        Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v184 = Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v47, v54);
        // pto: %14
        ;
        uint64_t v185 = (uint64_t) v38;
        TASSIGN(v184, v185);
        // pto: %56
        ;
        int64_t v186 = (int64_t) ((uint64_t) v173 + (uint64_t) v54);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
        uint16_t v187 = static_cast<uint16_t>(v49);
        uint16_t v188 = static_cast<uint16_t>(v186);
        TEXTRACT(v184, v170, v187, v188);
        int64_t v189 = (int64_t) ((uint64_t) v33 - (uint64_t) v173);
        // pto: %59
        ;
        int64_t v190 = v189 < v49 ? v49 : v189;
        // pto: %60
        ;
        int64_t v191 = v190 < v54 ? v190 : v54;
        // pto: %15
        ;
        Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v192 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v191, v44);
        // pto: %15
        ;
        uint64_t v193 = (uint64_t) v41;
        TASSIGN(v192, v193);
        uint16_t v194 = static_cast<uint16_t>(v186);
        uint16_t v195 = static_cast<uint16_t>(v49);
        TEXTRACT(v192, v171, v194, v195);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v73, v73, v174, v180);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
        TMATMUL_ACC(v73, v73, v184, v192);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      TFREE<TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>, TileSplitAxis::TILE_UP_DOWN>(v76);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    // pto: %62
    ;
    int64_t v196 = v80 < v49 ? v49 : v80;
    // pto: %mixes_partials_inline1438__iter_v1_pview
    ;
    __gm__ float* v197 = PTOAS__GLOBAL_TENSOR_DATA(v64);
    // pto: %mixes_partials_inline1438__iter_v1_pview
    ;
    const int64_t v198 = 0;
    // pto: %mixes_partials_inline1438__iter_v1_pview
    ;
    const int64_t v199 = 32;
    // pto: %mixes_partials_inline1438__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 48, 32> v200 = pto::Shape<1, 1, 1, 48, 32>();
    // pto: %mixes_partials_inline1438__iter_v1_pview
    ;
    pto::Stride<1536, 1536, 1536, 32, 1> v201 = pto::Stride<1536, 1536, 1536, 32, 1>();
    // pto: %mixes_partials_inline1438__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 48, 32>, pto::Stride<1536, 1536, 1536, 32, 1>, pto::Layout::ND> v202 = GlobalTensor<float, pto::Shape<1, 1, 1, 48, 32>, pto::Stride<1536, 1536, 1536, 32, 1>, pto::Layout::ND>(v197 + (v198 + v196 * v199), v200, v201);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v202, v81);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
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

AICORE void hc_large144_aiv(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ bfloat16_t* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, __gm__ float* v10, __gm__ float* v11, __gm__ bfloat16_t* v12, __gm__ float* v13, int64_t v14, int32_t v15, int64_t v16, int64_t v17, int64_t v18, int64_t v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int64_t v28, int64_t v29, int64_t v30, int32_t v31, int32_t v32, int32_t v33) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int64_t v34 = 12800;
  const int64_t v35 = 8704;
  const int64_t v36 = 4608;
  // pto: %c131200_i64
  const int64_t v37 = 131200;
  // pto: %c82048_i64
  const int64_t v38 = 82048;
  // pto: %c0_i64
  const int64_t v39 = 0;
  // pto: %c24576_i64
  const int64_t v40 = 24576;
  // pto: %c131296_i64
  const int64_t v41 = 131296;
  // pto: %c32768_i64
  const int64_t v42 = 32768;
  // pto: %c25600_i64
  const int64_t v43 = 25600;
  // pto: %c25856_i64
  const int64_t v44 = 25856;
  // pto: %c26112_i64
  const int64_t v45 = 26112;
  const int64_t v46 = 25632;
  const int64_t v47 = 24608;
  // pto: %c25088_i64
  const int64_t v48 = 25088;
  // pto: %c25344_i64
  const int64_t v49 = 25344;
  // pto: %c26368_i64
  const int64_t v50 = 26368;
  // pto: %c26880_i64
  const int64_t v51 = 26880;
  // pto: %c27392_i64
  const int64_t v52 = 27392;
  // pto: %c27904_i64
  const int64_t v53 = 27904;
  // pto: %c28416_i64
  const int64_t v54 = 28416;
  // pto: %c28928_i64
  const int64_t v55 = 28928;
  // pto: %c26400_i64
  const int64_t v56 = 26400;
  // pto: %c26432_i64
  const int64_t v57 = 26432;
  // pto: %c26464_i64
  const int64_t v58 = 26464;
  const int64_t v59 = 25664;
  // pto: %c24832_i64
  const int64_t v60 = 24832;
  // pto: %c24704_i64
  const int64_t v61 = 24704;
  // pto: %c73728_i64
  const int64_t v62 = 73728;
  // pto: %c73760_i64
  const int64_t v63 = 73760;
  // pto: %c73792_i64
  const int64_t v64 = 73792;
  const int64_t v65 = 25696;
  // pto: %c73824_i64
  const int64_t v66 = 73824;
  // pto: %c40960_i64
  const int64_t v67 = 40960;
  // pto: %c57344_i64
  const int64_t v68 = 57344;
  // pto: %c16384_i64
  const int64_t v69 = 16384;
  // pto: %c114816_i64
  const int64_t v70 = 114816;
  // pto: %c73856_i64
  const int64_t v71 = 73856;
  // pto: %c98432_i64
  const int64_t v72 = 98432;
  // pto: %c32_index
  const int64_t v73 = 32;
  // pto: %c1_index
  const int64_t v74 = 1;
  // pto: %c24_index
  const int64_t v75 = 24;
  // pto: %c16384_index
  const int64_t v76 = 16384;
  // pto: %c3_index
  const int64_t v77 = 3;
  // pto: %c8_index
  const int64_t v78 = 8;
  // pto: %c16_index
  const int64_t v79 = 16;
  // pto: %c4096_index
  const int64_t v80 = 4096;
  // pto: %c0_i32
  const int32_t v81 = 0;
  // pto: %c48_index
  const int64_t v82 = 48;
  // pto: %c2048_index
  const int64_t v83 = 2048;
  // pto: %cst_47
  const float v84 = 0.0f;
  // pto: %c0_index
  const int64_t v85 = 0;
  // pto: %c4_index
  const int64_t v86 = 4;
  // pto: %c2_index
  const int64_t v87 = 2;
  // pto: %c512_index
  const int64_t v88 = 512;
  // pto: %c256_index
  const int64_t v89 = 256;
  // pto: %cst_53
  const float v90 = 6.10351563E-5f;
  // pto: %cst_54
  const float v91 = 9.99999997E-7f;
  // pto: %c128_index
  const int64_t v92 = 128;
  // pto: %c192_index
  const int64_t v93 = 192;
  // pto: %cst_57
  const float v94 = 1.0f;
  // pto: %cst_58
  const float v95 = 0.5f;
  // pto: %cst_59
  const float v96 = 2.0f;
  // pto: %cst_60
  const float v97 = 0.25f;
  // pto: %cst_61
  const float v98 = 4.0f;
  // pto: %cst_62
  const float v99 = 0.125f;
  // pto: %cst_63
  const float v100 = 8.0f;
  // pto: %cst_64
  const float v101 = 0.0625f;
  // pto: %cst_65
  const float v102 = 16.0f;
  // pto: %c19_index
  const int64_t v103 = 19;
  // pto: %c64_index
  const int64_t v104 = 64;
  // pto: %c8192_index
  const int64_t v105 = 8192;
  // pto: %c12288_index
  const int64_t v106 = 12288;
  // pto: %237
  int64_t v107 = (int64_t) ((uint64_t) v29 * (uint64_t) v78);
  // pto: %rms_partials_inline1477__ssa_v0_view
  const int64_t v108 = 1;
  // pto: %rms_partials_inline1477__ssa_v0_view
  const int64_t v109 = 1;
  // pto: %rms_partials_inline1477__ssa_v0_view
  const int64_t v110 = 1;
  // pto: %rms_partials_inline1477__ssa_v0_view
  int64_t v111 = v107 * v74;
  // pto: %rms_partials_inline1477__ssa_v0_view
  int64_t v112 = v110 * v111;
  // pto: %rms_partials_inline1477__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v113 = pto::Shape<1, 1, 1, -1, -1>(v108, v109, v110, v107, v74);
  // pto: %rms_partials_inline1477__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v114 = pto::Stride<-1, -1, -1, -1, -1>(v109 * v112, v112, v111, v74, v107);
  // pto: %rms_partials_inline1477__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v115 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v113, v114);
  // pto: %x_flat_inline1417__ssa_v0_view
  const int64_t v116 = 1;
  // pto: %x_flat_inline1417__ssa_v0_view
  const int64_t v117 = 1;
  // pto: %x_flat_inline1417__ssa_v0_view
  const int64_t v118 = 1;
  // pto: %x_flat_inline1417__ssa_v0_view
  int64_t v119 = (int64_t) v30;
  // pto: %x_flat_inline1417__ssa_v0_view
  int64_t v120 = v119 * v76;
  // pto: %x_flat_inline1417__ssa_v0_view
  int64_t v121 = v118 * v120;
  // pto: %x_flat_inline1417__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v122 = pto::Shape<1, 1, 1, -1, -1>(v116, v117, v118, v119, v76);
  // pto: %x_flat_inline1417__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v123 = pto::Stride<-1, -1, -1, -1, -1>(v117 * v121, v121, v120, v76, v74);
  // pto: %x_flat_inline1417__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v124 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v122, v123);
  // pto: %238
  int64_t v125 = (int64_t) ((uint64_t) v29 * (uint64_t) v73);
  // pto: %parts_2d_inline1399__ssa_v0_view
  const int64_t v126 = 1;
  // pto: %parts_2d_inline1399__ssa_v0_view
  const int64_t v127 = 1;
  // pto: %parts_2d_inline1399__ssa_v0_view
  const int64_t v128 = 1;
  // pto: %parts_2d_inline1399__ssa_v0_view
  int64_t v129 = v78 * v125;
  // pto: %parts_2d_inline1399__ssa_v0_view
  int64_t v130 = v128 * v129;
  // pto: %parts_2d_inline1399__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v131 = pto::Shape<1, 1, 1, -1, -1>(v126, v127, v128, v78, v125);
  // pto: %parts_2d_inline1399__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v132 = pto::Stride<-1, -1, -1, -1, -1>(v127 * v130, v130, v129, v125, v74);
  // pto: %parts_2d_inline1399__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v133 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v131, v132);
  // pto: %rms_2d_inline1518__ssa_v0_view
  const int64_t v134 = 1;
  // pto: %rms_2d_inline1518__ssa_v0_view
  const int64_t v135 = 1;
  // pto: %rms_2d_inline1518__ssa_v0_view
  const int64_t v136 = 1;
  // pto: %rms_2d_inline1518__ssa_v0_view
  int64_t v137 = (int64_t) v29;
  // pto: %rms_2d_inline1518__ssa_v0_view
  int64_t v138 = v78 * v137;
  // pto: %rms_2d_inline1518__ssa_v0_view
  int64_t v139 = v136 * v138;
  // pto: %rms_2d_inline1518__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v140 = pto::Shape<1, 1, 1, -1, -1>(v134, v135, v136, v78, (int64_t) v29);
  // pto: %rms_2d_inline1518__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v141 = pto::Stride<-1, -1, -1, -1, -1>(v135 * v139, v139, v138, v137, v74);
  // pto: %rms_2d_inline1518__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v142 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v140, v141);
  // pto: %hc_base_2d_inline1486__ssa_v0_view
  const int64_t v143 = 1;
  // pto: %hc_base_2d_inline1486__ssa_v0_view
  const int64_t v144 = 1;
  // pto: %hc_base_2d_inline1486__ssa_v0_view
  const int64_t v145 = 1;
  // pto: %hc_base_2d_inline1486__ssa_v0_view
  int64_t v146 = v74 * v75;
  // pto: %hc_base_2d_inline1486__ssa_v0_view
  int64_t v147 = v145 * v146;
  // pto: %hc_base_2d_inline1486__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v148 = pto::Shape<1, 1, 1, -1, -1>(v143, v144, v145, v74, v75);
  // pto: %hc_base_2d_inline1486__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v149 = pto::Stride<-1, -1, -1, -1, -1>(v144 * v147, v147, v146, v75, v74);
  // pto: %hc_base_2d_inline1486__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v150 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v8, v148, v149);
  // pto: %comb_inline1434__ssa_v0_view
  const int64_t v151 = 1;
  // pto: %comb_inline1434__ssa_v0_view
  const int64_t v152 = 1;
  // pto: %comb_inline1434__ssa_v0_view
  const int64_t v153 = 1;
  // pto: %comb_inline1434__ssa_v0_view
  int64_t v154 = (int64_t) v30;
  // pto: %comb_inline1434__ssa_v0_view
  int64_t v155 = v154 * v79;
  // pto: %comb_inline1434__ssa_v0_view
  int64_t v156 = v153 * v155;
  // pto: %comb_inline1434__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v157 = pto::Shape<1, 1, 1, -1, -1>(v151, v152, v153, v154, v79);
  // pto: %comb_inline1434__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v158 = pto::Stride<-1, -1, -1, -1, -1>(v152 * v156, v156, v155, v79, v74);
  // pto: %comb_inline1434__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v159 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v9, v157, v158);
  // pto: %hc_scale_2d_inline1448__ssa_v0_view
  const int64_t v160 = 1;
  // pto: %hc_scale_2d_inline1448__ssa_v0_view
  const int64_t v161 = 1;
  // pto: %hc_scale_2d_inline1448__ssa_v0_view
  const int64_t v162 = 1;
  // pto: %hc_scale_2d_inline1448__ssa_v0_view
  int64_t v163 = v74 * v77;
  // pto: %hc_scale_2d_inline1448__ssa_v0_view
  int64_t v164 = v162 * v163;
  // pto: %hc_scale_2d_inline1448__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v165 = pto::Shape<1, 1, 1, -1, -1>(v160, v161, v162, v74, v77);
  // pto: %hc_scale_2d_inline1448__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v166 = pto::Stride<-1, -1, -1, -1, -1>(v161 * v164, v164, v163, v77, v74);
  // pto: %hc_scale_2d_inline1448__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v167 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v10, v165, v166);
  // pto: %239
  int64_t v168 = (int64_t) ((uint64_t) v30 * (uint64_t) v86);
  // pto: %post_flat_inline1440__ssa_v0_view
  const int64_t v169 = 1;
  // pto: %post_flat_inline1440__ssa_v0_view
  const int64_t v170 = 1;
  // pto: %post_flat_inline1440__ssa_v0_view
  const int64_t v171 = 1;
  // pto: %post_flat_inline1440__ssa_v0_view
  int64_t v172 = v74 * v168;
  // pto: %post_flat_inline1440__ssa_v0_view
  int64_t v173 = v171 * v172;
  // pto: %post_flat_inline1440__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v174 = pto::Shape<1, 1, 1, -1, -1>(v169, v170, v171, v74, v168);
  // pto: %post_flat_inline1440__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v175 = pto::Stride<-1, -1, -1, -1, -1>(v170 * v173, v173, v172, v168, v74);
  // pto: %post_flat_inline1440__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v176 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v11, v174, v175);
  // pto: %mixed_view_inline1415__ssa_v0_view
  const int64_t v177 = 1;
  // pto: %mixed_view_inline1415__ssa_v0_view
  const int64_t v178 = 1;
  // pto: %mixed_view_inline1415__ssa_v0_view
  const int64_t v179 = 1;
  // pto: %mixed_view_inline1415__ssa_v0_view
  int64_t v180 = (int64_t) v30;
  // pto: %mixed_view_inline1415__ssa_v0_view
  int64_t v181 = v180 * v80;
  // pto: %mixed_view_inline1415__ssa_v0_view
  int64_t v182 = v179 * v181;
  // pto: %mixed_view_inline1415__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v183 = pto::Shape<1, 1, 1, -1, -1>(v177, v178, v179, v180, v80);
  // pto: %mixed_view_inline1415__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v184 = pto::Stride<-1, -1, -1, -1, -1>(v178 * v182, v182, v181, v80, v74);
  // pto: %mixed_view_inline1415__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v185 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v12, v183, v184);
  const int32_t v186 = 0;
  auto v187 = TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>(v13, v186, v81);
  // pto: %worker_inline1431__ssa_v0
  int64_t v188 = (int64_t) v31;
  // pto: %linear_workers_inline1457__ssa_v0_idx
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
  for (int64_t v189 = v188; v189 < v14; v189 += (int64_t) v15) {
    // pto: %48, %49
    ;
    int64_t v190 = (int64_t) ((uint64_t) (v189 / v78) * (uint64_t) v82);
    // pto: %50
    ;
    int64_t v191 = v189 % v78;
    // pto: %51
    ;
    int64_t v192 = (int64_t) ((uint64_t) v191 * (uint64_t) v83);
    // pto: %mid_stream_rms_total_inline1354__tile
    ;
    Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v193 = Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v75);
    // pto: %mid_stream_rms_total_inline1354__tile
    ;
    uint64_t v194 = (uint64_t) v37;
    TASSIGN(v193, v194);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TEXPANDS(v193, v84);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    for (int64_t v195 = v85; v195 < v86; v195 += v87) {
      // pto: %54
      ;
      int64_t v196 = (int64_t) ((uint64_t) v195 * (uint64_t) v88);
      // pto: %55
      ;
      int64_t v197 = (int64_t) ((uint64_t) v192 + (uint64_t) v196);
      // pto: %57, %lane_inline1488__ssa_v0, %56
      ;
      int64_t v198 = (int64_t) ((uint64_t) v190 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) v33) * (uint64_t) v75));
      // pto: %58
      ;
      int64_t v199 = (int64_t) ((uint64_t) v17 - (uint64_t) v198);
      // pto: %59
      ;
      int64_t v200 = v199 < v75 ? v199 : v75;
      // pto: %60
      ;
      int64_t v201 = v200 < v85 ? v85 : v200;
      // pto: %63, %62
      ;
      int64_t v202 = (int64_t) ((uint64_t) v192 + (uint64_t) ((int64_t) (uint64_t) v196 + (uint64_t) v88));
      // pto: %mid_stream_source_inline1368__tile
      ;
      Tile<TileType::Vec, bfloat16_t, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v203 = Tile<TileType::Vec, bfloat16_t, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v201, v88);
      // pto: %mid_stream_source_inline1368__tile
      ;
      uint64_t v204 = (uint64_t) v38;
      TASSIGN(v203, v204);
      // pto: %70
      ;
      int64_t v205 = v198 < v85 ? v85 : v198;
      // pto: %71
      ;
      int64_t v206 = v197 < v85 ? v85 : v197;
      // pto: %x_flat_inline1417__ssa_v0_pview
      ;
      const int64_t v207 = 0;
      // pto: %x_flat_inline1417__ssa_v0_pview
      ;
      __gm__ bfloat16_t* v208 = PTOAS__GLOBAL_TENSOR_DATA(v124);
      // pto: %x_flat_inline1417__ssa_v0_pview
      ;
      const int64_t v209 = 1;
      // pto: %x_flat_inline1417__ssa_v0_pview
      ;
      const int64_t v210 = 1;
      // pto: %x_flat_inline1417__ssa_v0_pview
      ;
      const int64_t v211 = 1;
      // pto: %x_flat_inline1417__ssa_v0_pview
      ;
      int64_t v212 = v201 * v76;
      // pto: %x_flat_inline1417__ssa_v0_pview
      ;
      int64_t v213 = v211 * v212;
      // pto: %x_flat_inline1417__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, -1> v214 = pto::Shape<1, 1, 1, -1, -1>(v209, v210, v211, v201, v88);
      // pto: %x_flat_inline1417__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v215 = pto::Stride<-1, -1, -1, -1, -1>(v210 * v213, v213, v212, v76, v74);
      // pto: %x_flat_inline1417__ssa_v0_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v216 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v208 + (v207 + v205 * v76 + v206 * v74), v214, v215);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v203, v216);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %0
      ;
      Tile<TileType::Vec, bfloat16_t, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v217 = Tile<TileType::Vec, bfloat16_t, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v201, v88);
      // pto: %0
      ;
      uint64_t v218 = (uint64_t) v39;
      TASSIGN(v217, v218);
      // pto: %73
      ;
      int64_t v219 = v202 < v85 ? v85 : v202;
      // pto: %74
      ;
      const int64_t v220 = 0;
      // pto: %74
      ;
      __gm__ bfloat16_t* v221 = PTOAS__GLOBAL_TENSOR_DATA(v124);
      // pto: %74
      ;
      const int64_t v222 = 1;
      // pto: %74
      ;
      const int64_t v223 = 1;
      // pto: %74
      ;
      const int64_t v224 = 1;
      // pto: %74
      ;
      int64_t v225 = v201 * v76;
      // pto: %74
      ;
      int64_t v226 = v224 * v225;
      // pto: %74
      ;
      pto::Shape<1, 1, 1, -1, -1> v227 = pto::Shape<1, 1, 1, -1, -1>(v222, v223, v224, v201, v88);
      // pto: %74
      ;
      pto::Stride<-1, -1, -1, -1, -1> v228 = pto::Stride<-1, -1, -1, -1, -1>(v223 * v226, v226, v225, v76, v74);
      // pto: %74
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v229 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v221 + (v220 + v205 * v76 + v219 * v74), v227, v228);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      TLOAD(v217, v229);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %mid_stream_widened_inline1412__tile
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v230 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v201, v88);
      // pto: %mid_stream_widened_inline1412__tile
      ;
      uint64_t v231 = (uint64_t) v40;
      TASSIGN(v230, v231);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      RoundMode v232 = RoundMode::CAST_ROUND;
      SaturationMode v233 = SaturationMode::ON;
      TCVT(v230, v203, v232, v233);
      v230.SetValidShape(v201, v88);
      // pto: %mid_stream_clean_inline1365__tile
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v234 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v75, v88);
      // pto: %mid_stream_clean_inline1365__tile
      ;
      uint64_t v235 = (uint64_t) v40;
      TASSIGN(v234, v235);
      pipe_barrier(PIPE_V);
      TFILLPAD<pto::TFillPadMode::InPlace>(v234, v230);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      v234.SetValidShape(v75, v88);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TPUSH<TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>, Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v187, v234);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      // pto: %1
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v236 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v75, v88);
      // pto: %1
      ;
      uint64_t v237 = (uint64_t) v40;
      TASSIGN(v236, v237);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      TMUL(v236, v234, v234);
      // pto: %tmp_tile
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v238 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v75, v88);
      // pto: %tmp_tile
      ;
      uint64_t v239 = (uint64_t) v38;
      TASSIGN(v238, v239);
      // pto: %2
      ;
      Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v240 = Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v75, v74);
      // pto: %2
      ;
      uint64_t v241 = (uint64_t) v41;
      TASSIGN(v240, v241);
      pipe_barrier(PIPE_V);
      TROWSUM(v240, v236, v238);
      // pto: %mid_stream_sq_inline1385__tile
      ;
      Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v242 = Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v75);
      // pto: %mid_stream_sq_inline1385__tile
      ;
      uint64_t v243 = (uint64_t) v41;
      TASSIGN(v242, v243);
      // pto: %3
      ;
      Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v244 = Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v75);
      // pto: %3
      ;
      uint64_t v245 = (uint64_t) v41;
      TASSIGN(v244, v245);
      pipe_barrier(PIPE_V);
      TADD(v244, v193, v242);
      // pto: %4
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v246 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v201, v88);
      // pto: %4
      ;
      uint64_t v247 = (uint64_t) v40;
      TASSIGN(v246, v247);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      RoundMode v248 = RoundMode::CAST_ROUND;
      SaturationMode v249 = SaturationMode::ON;
      TCVT(v246, v217, v248, v249);
      v246.SetValidShape(v201, v88);
      // pto: %6
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v250 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v75, v88);
      // pto: %6
      ;
      uint64_t v251 = (uint64_t) v40;
      TASSIGN(v250, v251);
      pipe_barrier(PIPE_V);
      TFILLPAD<pto::TFillPadMode::InPlace>(v250, v246);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      v250.SetValidShape(v75, v88);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      TPUSH<TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>, Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v187, v250);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      // pto: %8
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v252 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v75, v88);
      // pto: %8
      ;
      uint64_t v253 = (uint64_t) v40;
      TASSIGN(v252, v253);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      TMUL(v252, v250, v250);
      // pto: %9
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v254 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v75, v88);
      // pto: %9
      ;
      uint64_t v255 = (uint64_t) v38;
      TASSIGN(v254, v255);
      // pto: %10
      ;
      Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v256 = Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v75, v74);
      // pto: %10
      ;
      uint64_t v257 = (uint64_t) v39;
      TASSIGN(v256, v257);
      pipe_barrier(PIPE_V);
      TROWSUM(v256, v252, v254);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      // pto: %11
      ;
      Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v258 = Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v75);
      // pto: %11
      ;
      uint64_t v259 = (uint64_t) v39;
      TASSIGN(v258, v259);
      // pto: %12
      ;
      Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v260 = Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v75);
      // pto: %12
      ;
      uint64_t v261 = (uint64_t) v37;
      TASSIGN(v260, v261);
      pipe_barrier(PIPE_V);
      TADD(v260, v244, v258);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    };
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    // pto: %52, %53, %76, %rms_lane_inline1481__ssa_v0, %75
    ;
    int64_t v262 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v191 * (uint64_t) v16) + (uint64_t) v190) + (uint64_t) ((int64_t) (uint64_t) ((int64_t) v33) * (uint64_t) v75));
    // pto: %13
    ;
    Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v263 = Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v75, v74);
    // pto: %13
    ;
    uint64_t v264 = (uint64_t) v37;
    TASSIGN(v263, v264);
    // pto: %77
    ;
    int64_t v265 = v262 < v85 ? v85 : v262;
    // pto: %rms_partials_inline1477__iter_v1_pview
    ;
    const int64_t v266 = 0;
    // pto: %rms_partials_inline1477__iter_v1_pview
    ;
    __gm__ float* v267 = PTOAS__GLOBAL_TENSOR_DATA(v115);
    // pto: %rms_partials_inline1477__iter_v1_pview
    ;
    const int64_t v268 = 1;
    // pto: %rms_partials_inline1477__iter_v1_pview
    ;
    const int64_t v269 = 1;
    // pto: %rms_partials_inline1477__iter_v1_pview
    ;
    const int64_t v270 = 1;
    // pto: %rms_partials_inline1477__iter_v1_pview
    ;
    int64_t v271 = v75 * v74;
    // pto: %rms_partials_inline1477__iter_v1_pview
    ;
    int64_t v272 = v270 * v271;
    // pto: %rms_partials_inline1477__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 24, 1> v273 = pto::Shape<1, 1, 1, 24, 1>(v268, v269, v270, v75, v74);
    // pto: %rms_partials_inline1477__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v274 = pto::Stride<-1, -1, -1, -1, -1>(v269 * v272, v272, v271, v74, v107);
    // pto: %rms_partials_inline1477__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 24, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v275 = GlobalTensor<float, pto::Shape<1, 1, 1, 24, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v267 + (v266 + v265 * v74 + v85 * v107), v273, v274);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    TSTORE(v275, v263);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  pipe_barrier(PIPE_ALL);
  dsb(DSB_DDR);
  SYNCALL<SyncCoreType::Mix>();
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  // pto: %78, %79, %tail_lane_inline1437__ssa_v0
  int64_t v276 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v188 * (uint64_t) v87) + (uint64_t) ((int64_t) v33));
  // pto: %80
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  if (v276 < v18) {
    // pto: %82
    ;
    int64_t v277 = v276 < v20 ? v276 : v20;
    // pto: %81, %83
    ;
    int64_t v278 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v276 * (uint64_t) v19) + (uint64_t) v277);
    // pto: %85
    ;
    int64_t v279 = (int64_t) ((uint64_t) v20 - (uint64_t) v276);
    // pto: %86
    ;
    int64_t v280 = v279 < v85 ? v85 : v279;
    // pto: %87
    ;
    int64_t v281 = v280 < v74 ? v280 : v74;
    // pto: %84, %88
    ;
    int64_t v282 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v278 + (uint64_t) v19) + (uint64_t) v281);
    for (int64_t v283 = v278; v283 < v282; v283 += v78) {
      // pto: %scale2_inline1374__tile
      ;
      float v284 = (v5)[v87];
      // pto: %89
      ;
      int64_t v285 = (int64_t) ((uint64_t) v282 - (uint64_t) v283);
      // pto: %90
      ;
      int64_t v286 = v285 < v78 ? v285 : v78;
      // pto: %91
      ;
      int64_t v287 = (int64_t) ((uint64_t) v286 * (uint64_t) v73);
      // pto: %c_parts_load_inline1403__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v288 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v287);
      // pto: %c_parts_load_inline1403__ssa_v0
      ;
      uint64_t v289 = (uint64_t) v40;
      TASSIGN(v288, v289);
      // pto: %92
      ;
      int64_t v290 = (int64_t) ((uint64_t) v283 * (uint64_t) v73);
      // pto: %93
      ;
      int64_t v291 = v290 < v85 ? v85 : v290;
      // pto: %parts_2d_inline1399__ssa_v0_pview
      ;
      const int64_t v292 = 0;
      // pto: %parts_2d_inline1399__ssa_v0_pview
      ;
      __gm__ float* v293 = PTOAS__GLOBAL_TENSOR_DATA(v133);
      // pto: %parts_2d_inline1399__ssa_v0_pview
      ;
      const int64_t v294 = 1;
      // pto: %parts_2d_inline1399__ssa_v0_pview
      ;
      const int64_t v295 = 1;
      // pto: %parts_2d_inline1399__ssa_v0_pview
      ;
      const int64_t v296 = 1;
      // pto: %parts_2d_inline1399__ssa_v0_pview
      ;
      int64_t v297 = v78 * v125;
      // pto: %parts_2d_inline1399__ssa_v0_pview
      ;
      int64_t v298 = v296 * v297;
      // pto: %parts_2d_inline1399__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, -1> v299 = pto::Shape<1, 1, 1, 8, -1>(v294, v295, v296, v78, v287);
      // pto: %parts_2d_inline1399__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v300 = pto::Stride<-1, -1, -1, -1, -1>(v295 * v298, v298, v297, v125, v74);
      // pto: %parts_2d_inline1399__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v301 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v293 + (v292 + v85 * v125 + v291 * v74), v299, v300);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      TLOAD(v288, v301);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %t__tmp_v306
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v302 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v78, v89);
      // pto: %t__tmp_v306
      ;
      uint64_t v303 = (uint64_t) v40;
      TASSIGN(v302, v303);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      TFILLPAD<pto::TFillPadMode::InPlace>(v302, v288);
      v302.SetValidShape(v78, v89);
      // pto: %c_parts_tmp_inline1340__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v304 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v89);
      // pto: %c_parts_tmp_inline1340__ssa_v0
      ;
      uint64_t v305 = (uint64_t) v42;
      TASSIGN(v304, v305);
      // pto: %t__tmp_v307
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v306 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v89);
      // pto: %t__tmp_v307
      ;
      uint64_t v307 = (uint64_t) v40;
      TASSIGN(v306, v307);
      // pto: %colsum_src_view
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, 8, 256, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v308;
      TRESHAPE(v308, v302);
      // pto: %colsum_tmp_view
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, 8, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v309;
      TRESHAPE(v309, v304);
      // pto: %colsum_dst_view
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v310;
      TRESHAPE(v310, v306);
      pipe_barrier(PIPE_V);
      const bool v311 = true;
      TCOLSUM(v310, v308, v309, v311);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      // pto: %c_total_inline1383__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v312 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v73);
      // pto: %c_total_inline1383__ssa_v0
      ;
      uint64_t v313 = (uint64_t) v40;
      TASSIGN(v312, v313);
      // pto: %c_sq_load_inline1482__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v314 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v286);
      // pto: %c_sq_load_inline1482__ssa_v0
      ;
      uint64_t v315 = (uint64_t) v43;
      TASSIGN(v314, v315);
      // pto: %95
      ;
      int64_t v316 = v283 < v85 ? v85 : v283;
      // pto: %rms_2d_inline1518__ssa_v0_pview
      ;
      int64_t v317 = (int64_t) v29;
      // pto: %rms_2d_inline1518__ssa_v0_pview
      ;
      const int64_t v318 = 0;
      // pto: %rms_2d_inline1518__ssa_v0_pview
      ;
      __gm__ float* v319 = PTOAS__GLOBAL_TENSOR_DATA(v142);
      // pto: %rms_2d_inline1518__ssa_v0_pview
      ;
      const int64_t v320 = 1;
      // pto: %rms_2d_inline1518__ssa_v0_pview
      ;
      const int64_t v321 = 1;
      // pto: %rms_2d_inline1518__ssa_v0_pview
      ;
      const int64_t v322 = 1;
      // pto: %rms_2d_inline1518__ssa_v0_pview
      ;
      int64_t v323 = v78 * v317;
      // pto: %rms_2d_inline1518__ssa_v0_pview
      ;
      int64_t v324 = v322 * v323;
      // pto: %rms_2d_inline1518__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, -1> v325 = pto::Shape<1, 1, 1, 8, -1>(v320, v321, v322, v78, v286);
      // pto: %rms_2d_inline1518__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v326 = pto::Stride<-1, -1, -1, -1, -1>(v321 * v324, v324, v323, v317, v74);
      // pto: %rms_2d_inline1518__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v327 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v319 + (v318 + v85 * v317 + v316 * v74), v325, v326);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      TLOAD(v314, v327);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      // pto: %t__tmp_v308
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v328 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v78, v78);
      // pto: %t__tmp_v308
      ;
      uint64_t v329 = (uint64_t) v43;
      TASSIGN(v328, v329);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      TFILLPAD<pto::TFillPadMode::InPlace>(v328, v314);
      v328.SetValidShape(v78, v78);
      // pto: %c_sq_tmp_inline1337__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v330 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %c_sq_tmp_inline1337__ssa_v0
      ;
      uint64_t v331 = (uint64_t) v44;
      TASSIGN(v330, v331);
      // pto: %c_rms_tmp_inline1407__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v332 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %c_rms_tmp_inline1407__ssa_v0
      ;
      uint64_t v333 = (uint64_t) v45;
      TASSIGN(v332, v333);
      // pto: %t__tmp_v309
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v334 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %t__tmp_v309
      ;
      uint64_t v335 = (uint64_t) v43;
      TASSIGN(v334, v335);
      // pto: %96
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v336;
      TRESHAPE(v336, v328);
      // pto: %97
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v337;
      TRESHAPE(v337, v330);
      // pto: %98
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v338;
      TRESHAPE(v338, v334);
      pipe_barrier(PIPE_V);
      const bool v339 = true;
      TCOLSUM(v338, v336, v337, v339);
      // pto: %t__tmp_v310
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v340 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %t__tmp_v310
      ;
      uint64_t v341 = (uint64_t) v43;
      TASSIGN(v340, v341);
      pipe_barrier(PIPE_V);
      TMULS(v340, v334, v90);
      // pto: %t__tmp_v311
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v342 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %t__tmp_v311
      ;
      uint64_t v343 = (uint64_t) v43;
      TASSIGN(v342, v343);
      pipe_barrier(PIPE_V);
      TADDS(v342, v340, v91);
      // pto: %t__tmp_v312
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v344 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %t__tmp_v312
      ;
      uint64_t v345 = (uint64_t) v46;
      TASSIGN(v344, v345);
      pipe_barrier(PIPE_V);
      TRSQRT(v344, v342, v332);
      // pto: %c_inv_inline1336__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v346 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v74);
      // pto: %c_inv_inline1336__ssa_v0
      ;
      uint64_t v347 = (uint64_t) v46;
      TASSIGN(v346, v347);
      // pto: %comb_inv_inline1494__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v348 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v74);
      // pto: %comb_inv_inline1494__ssa_v0
      ;
      uint64_t v349 = (uint64_t) v46;
      TASSIGN(v348, v349);
      // pto: %t__tmp_v313
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v350 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v79);
      // pto: %t__tmp_v313
      ;
      uint64_t v351 = (uint64_t) v47;
      TASSIGN(v350, v351);
      // pto: %slice_view
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 16, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v352;
      // pto: %slice_view
      ;
      uint64_t v353 = (uint64_t) v47;
      TASSIGN(v352, v353);
      // pto: %raw_inline1540__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v354 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v79);
      // pto: %raw_inline1540__ssa_v0
      ;
      uint64_t v355 = (uint64_t) v40;
      TASSIGN(v354, v355);
      TADDS(v354, v352, v84);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      // pto: %base_inline1464__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v356 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v79);
      // pto: %base_inline1464__ssa_v0
      ;
      uint64_t v357 = (uint64_t) v48;
      TASSIGN(v356, v357);
      // pto: %hc_base_2d_inline1486__ssa_v0_pview
      ;
      __gm__ float* v358 = PTOAS__GLOBAL_TENSOR_DATA(v150);
      // pto: %hc_base_2d_inline1486__ssa_v0_pview
      ;
      unsigned v359 = 8;
      // pto: %hc_base_2d_inline1486__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 16> v360 = pto::Shape<1, 1, 1, 1, 16>();
      // pto: %hc_base_2d_inline1486__ssa_v0_pview
      ;
      pto::Stride<24, 24, 24, 24, 1> v361 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %hc_base_2d_inline1486__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v362 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v358 + v359, v360, v361);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      TLOAD(v356, v362);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      // pto: %t__tmp_v314
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v363 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v79);
      // pto: %t__tmp_v314
      ;
      uint64_t v364 = (uint64_t) v40;
      TASSIGN(v363, v364);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v363, v354, v348);
      // pto: %t__tmp_v315
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v365 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v79);
      // pto: %t__tmp_v315
      ;
      uint64_t v366 = (uint64_t) v40;
      TASSIGN(v365, v366);
      pipe_barrier(PIPE_V);
      TMULS(v365, v363, v284);
      // pto: %logits_inline1497__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v367 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v79);
      // pto: %logits_inline1497__ssa_v0
      ;
      uint64_t v368 = (uint64_t) v49;
      TASSIGN(v367, v368);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      TCOLEXPANDADD(v367, v365, v356);
      // pto: %t__tmp_v316
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v369 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v79);
      // pto: %t__tmp_v316
      ;
      uint64_t v370 = (uint64_t) v49;
      TASSIGN(v369, v370);
      // pto: %regular_logits_inline1371__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v371 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %regular_logits_inline1371__ssa_v0
      ;
      uint64_t v372 = (uint64_t) v49;
      TASSIGN(v371, v372);
      // pto: %ids_inline1348__ci_tmp_v0
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v373 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v93);
      // pto: %ids_inline1348__ci_tmp_v0
      ;
      uint64_t v374 = (uint64_t) v40;
      TASSIGN(v373, v374);
      // pto: %ids_inline1348__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v375 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %ids_inline1348__ssa_v1
      ;
      uint64_t v376 = (uint64_t) v44;
      TASSIGN(v375, v376);
      // pto: %ci_tmp_view
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v377;
      TRESHAPE(v377, v373);
      // pto: %ci_dst_view
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v378;
      TRESHAPE(v378, v375);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v378, v81, v377);
      // pto: %tmp_inline1439__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v379 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %tmp_inline1439__ssa_v1
      ;
      uint64_t v380 = (uint64_t) v40;
      TASSIGN(v379, v380);
      // pto: %idx_float_inline1465__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v381 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %idx_float_inline1465__ssa_v0
      ;
      uint64_t v382 = (uint64_t) v44;
      TASSIGN(v381, v382);
      pipe_barrier(PIPE_V);
      RoundMode v383 = RoundMode::CAST_ROUND;
      SaturationMode v384 = SaturationMode::ON;
      TCVT(v381, v375, v383, v384);
      // pto: %t__tmp_v317
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v385 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v317
      ;
      uint64_t v386 = (uint64_t) v50;
      TASSIGN(v385, v386);
      pipe_barrier(PIPE_V);
      TMULS(v385, v381, v94);
      // pto: %t__tmp_v318
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v387 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v318
      ;
      uint64_t v388 = (uint64_t) v50;
      TASSIGN(v387, v388);
      pipe_barrier(PIPE_V);
      RoundMode v389 = RoundMode::CAST_FLOOR;
      SaturationMode v390 = SaturationMode::ON;
      TCVT(v387, v385, v389, v390);
      // pto: %ri1_low_inline1555__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v391 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %ri1_low_inline1555__ssa_v0
      ;
      uint64_t v392 = (uint64_t) v50;
      TASSIGN(v391, v392);
      pipe_barrier(PIPE_V);
      RoundMode v393 = RoundMode::CAST_ROUND;
      SaturationMode v394 = SaturationMode::ON;
      TCVT(v391, v387, v393, v394);
      // pto: %t__tmp_v319
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v395 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v319
      ;
      uint64_t v396 = (uint64_t) v51;
      TASSIGN(v395, v396);
      TMULS(v395, v381, v95);
      // pto: %t__tmp_v320
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v397 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v320
      ;
      uint64_t v398 = (uint64_t) v51;
      TASSIGN(v397, v398);
      pipe_barrier(PIPE_V);
      RoundMode v399 = RoundMode::CAST_FLOOR;
      SaturationMode v400 = SaturationMode::ON;
      TCVT(v397, v395, v399, v400);
      // pto: %ri1_high_inline1350__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v401 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %ri1_high_inline1350__ssa_v0
      ;
      uint64_t v402 = (uint64_t) v51;
      TASSIGN(v401, v402);
      pipe_barrier(PIPE_V);
      RoundMode v403 = RoundMode::CAST_ROUND;
      SaturationMode v404 = SaturationMode::ON;
      TCVT(v401, v397, v403, v404);
      // pto: %t__tmp_v321
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v405 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v321
      ;
      uint64_t v406 = (uint64_t) v51;
      TASSIGN(v405, v406);
      pipe_barrier(PIPE_V);
      TMULS(v405, v401, v96);
      // pto: %ri1_bit_inline1557__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v407 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %ri1_bit_inline1557__ssa_v0
      ;
      uint64_t v408 = (uint64_t) v50;
      TASSIGN(v407, v408);
      pipe_barrier(PIPE_V);
      TSUB(v407, v391, v405);
      // pto: %t__tmp_v322
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v409 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v322
      ;
      uint64_t v410 = (uint64_t) v50;
      TASSIGN(v409, v410);
      pipe_barrier(PIPE_V);
      TMULS(v409, v407, v96);
      // pto: %t__tmp_v323
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v411 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v323
      ;
      uint64_t v412 = (uint64_t) v50;
      TASSIGN(v411, v412);
      pipe_barrier(PIPE_V);
      TSUB(v411, v381, v409);
      // pto: %t__tmp_v324
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v413 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v324
      ;
      uint64_t v414 = (uint64_t) v50;
      TASSIGN(v413, v414);
      pipe_barrier(PIPE_V);
      TADDS(v413, v411, v94);
      // pto: %ri1_inline1519__ssa_v2
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v415 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %ri1_inline1519__ssa_v2
      ;
      uint64_t v416 = (uint64_t) v50;
      TASSIGN(v415, v416);
      pipe_barrier(PIPE_V);
      RoundMode v417 = RoundMode::CAST_ROUND;
      SaturationMode v418 = SaturationMode::ON;
      TCVT(v415, v413, v417, v418);
      // pto: %t__tmp_v325
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v419 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v325
      ;
      uint64_t v420 = (uint64_t) v51;
      TASSIGN(v419, v420);
      TMULS(v419, v381, v95);
      // pto: %t__tmp_v326
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v421 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v326
      ;
      uint64_t v422 = (uint64_t) v51;
      TASSIGN(v421, v422);
      pipe_barrier(PIPE_V);
      RoundMode v423 = RoundMode::CAST_FLOOR;
      SaturationMode v424 = SaturationMode::ON;
      TCVT(v421, v419, v423, v424);
      // pto: %ri2_low_inline1559__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v425 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %ri2_low_inline1559__ssa_v0
      ;
      uint64_t v426 = (uint64_t) v51;
      TASSIGN(v425, v426);
      pipe_barrier(PIPE_V);
      RoundMode v427 = RoundMode::CAST_ROUND;
      SaturationMode v428 = SaturationMode::ON;
      TCVT(v425, v421, v427, v428);
      // pto: %t__tmp_v327
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v429 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v327
      ;
      uint64_t v430 = (uint64_t) v52;
      TASSIGN(v429, v430);
      TMULS(v429, v381, v97);
      // pto: %t__tmp_v328
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v431 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v328
      ;
      uint64_t v432 = (uint64_t) v52;
      TASSIGN(v431, v432);
      pipe_barrier(PIPE_V);
      RoundMode v433 = RoundMode::CAST_FLOOR;
      SaturationMode v434 = SaturationMode::ON;
      TCVT(v431, v429, v433, v434);
      // pto: %ri2_high_inline1560__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v435 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %ri2_high_inline1560__ssa_v0
      ;
      uint64_t v436 = (uint64_t) v52;
      TASSIGN(v435, v436);
      pipe_barrier(PIPE_V);
      RoundMode v437 = RoundMode::CAST_ROUND;
      SaturationMode v438 = SaturationMode::ON;
      TCVT(v435, v431, v437, v438);
      // pto: %t__tmp_v329
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v439 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v329
      ;
      uint64_t v440 = (uint64_t) v52;
      TASSIGN(v439, v440);
      pipe_barrier(PIPE_V);
      TMULS(v439, v435, v96);
      // pto: %ri2_bit_inline1545__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v441 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %ri2_bit_inline1545__ssa_v0
      ;
      uint64_t v442 = (uint64_t) v51;
      TASSIGN(v441, v442);
      pipe_barrier(PIPE_V);
      TSUB(v441, v425, v439);
      // pto: %t__tmp_v330
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v443 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v330
      ;
      uint64_t v444 = (uint64_t) v51;
      TASSIGN(v443, v444);
      pipe_barrier(PIPE_V);
      TMULS(v443, v441, v98);
      // pto: %t__tmp_v331
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v445 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v331
      ;
      uint64_t v446 = (uint64_t) v51;
      TASSIGN(v445, v446);
      pipe_barrier(PIPE_V);
      TSUB(v445, v381, v443);
      // pto: %t__tmp_v332
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v447 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v332
      ;
      uint64_t v448 = (uint64_t) v51;
      TASSIGN(v447, v448);
      pipe_barrier(PIPE_V);
      TADDS(v447, v445, v96);
      // pto: %ri2_inline1551__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v449 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %ri2_inline1551__ssa_v1
      ;
      uint64_t v450 = (uint64_t) v51;
      TASSIGN(v449, v450);
      pipe_barrier(PIPE_V);
      RoundMode v451 = RoundMode::CAST_ROUND;
      SaturationMode v452 = SaturationMode::ON;
      TCVT(v449, v447, v451, v452);
      // pto: %t__tmp_v333
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v453 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v333
      ;
      uint64_t v454 = (uint64_t) v52;
      TASSIGN(v453, v454);
      TMULS(v453, v381, v97);
      // pto: %t__tmp_v334
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v455 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v334
      ;
      uint64_t v456 = (uint64_t) v52;
      TASSIGN(v455, v456);
      pipe_barrier(PIPE_V);
      RoundMode v457 = RoundMode::CAST_FLOOR;
      SaturationMode v458 = SaturationMode::ON;
      TCVT(v455, v453, v457, v458);
      // pto: %ci1_low_inline1561__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v459 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %ci1_low_inline1561__ssa_v0
      ;
      uint64_t v460 = (uint64_t) v52;
      TASSIGN(v459, v460);
      pipe_barrier(PIPE_V);
      RoundMode v461 = RoundMode::CAST_ROUND;
      SaturationMode v462 = SaturationMode::ON;
      TCVT(v459, v455, v461, v462);
      // pto: %t__tmp_v335
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v463 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v335
      ;
      uint64_t v464 = (uint64_t) v53;
      TASSIGN(v463, v464);
      TMULS(v463, v381, v99);
      // pto: %t__tmp_v336
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v465 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v336
      ;
      uint64_t v466 = (uint64_t) v53;
      TASSIGN(v465, v466);
      pipe_barrier(PIPE_V);
      RoundMode v467 = RoundMode::CAST_FLOOR;
      SaturationMode v468 = SaturationMode::ON;
      TCVT(v465, v463, v467, v468);
      // pto: %ci1_high_inline1562__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v469 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %ci1_high_inline1562__ssa_v0
      ;
      uint64_t v470 = (uint64_t) v53;
      TASSIGN(v469, v470);
      pipe_barrier(PIPE_V);
      RoundMode v471 = RoundMode::CAST_ROUND;
      SaturationMode v472 = SaturationMode::ON;
      TCVT(v469, v465, v471, v472);
      // pto: %t__tmp_v337
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v473 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v337
      ;
      uint64_t v474 = (uint64_t) v53;
      TASSIGN(v473, v474);
      pipe_barrier(PIPE_V);
      TMULS(v473, v469, v96);
      // pto: %ci1_bit_inline1564__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v475 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %ci1_bit_inline1564__ssa_v0
      ;
      uint64_t v476 = (uint64_t) v52;
      TASSIGN(v475, v476);
      pipe_barrier(PIPE_V);
      TSUB(v475, v459, v473);
      // pto: %t__tmp_v338
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v477 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v338
      ;
      uint64_t v478 = (uint64_t) v52;
      TASSIGN(v477, v478);
      pipe_barrier(PIPE_V);
      TMULS(v477, v475, v100);
      // pto: %t__tmp_v339
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v479 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v339
      ;
      uint64_t v480 = (uint64_t) v52;
      TASSIGN(v479, v480);
      pipe_barrier(PIPE_V);
      TSUB(v479, v381, v477);
      // pto: %t__tmp_v340
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v481 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v340
      ;
      uint64_t v482 = (uint64_t) v52;
      TASSIGN(v481, v482);
      pipe_barrier(PIPE_V);
      TADDS(v481, v479, v98);
      // pto: %ci1_inline1510__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v483 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %ci1_inline1510__ssa_v1
      ;
      uint64_t v484 = (uint64_t) v52;
      TASSIGN(v483, v484);
      pipe_barrier(PIPE_V);
      RoundMode v485 = RoundMode::CAST_ROUND;
      SaturationMode v486 = SaturationMode::ON;
      TCVT(v483, v481, v485, v486);
      // pto: %t__tmp_v341
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v487 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v341
      ;
      uint64_t v488 = (uint64_t) v53;
      TASSIGN(v487, v488);
      TMULS(v487, v381, v99);
      // pto: %t__tmp_v342
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v489 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v342
      ;
      uint64_t v490 = (uint64_t) v53;
      TASSIGN(v489, v490);
      pipe_barrier(PIPE_V);
      RoundMode v491 = RoundMode::CAST_FLOOR;
      SaturationMode v492 = SaturationMode::ON;
      TCVT(v489, v487, v491, v492);
      // pto: %ci2_low_inline1568__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v493 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %ci2_low_inline1568__ssa_v0
      ;
      uint64_t v494 = (uint64_t) v53;
      TASSIGN(v493, v494);
      pipe_barrier(PIPE_V);
      RoundMode v495 = RoundMode::CAST_ROUND;
      SaturationMode v496 = SaturationMode::ON;
      TCVT(v493, v489, v495, v496);
      // pto: %t__tmp_v343
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v497 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v343
      ;
      uint64_t v498 = (uint64_t) v54;
      TASSIGN(v497, v498);
      TMULS(v497, v381, v101);
      // pto: %t__tmp_v344
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v499 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v344
      ;
      uint64_t v500 = (uint64_t) v54;
      TASSIGN(v499, v500);
      pipe_barrier(PIPE_V);
      RoundMode v501 = RoundMode::CAST_FLOOR;
      SaturationMode v502 = SaturationMode::ON;
      TCVT(v499, v497, v501, v502);
      // pto: %ci2_high_inline1450__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v503 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %ci2_high_inline1450__ssa_v0
      ;
      uint64_t v504 = (uint64_t) v54;
      TASSIGN(v503, v504);
      pipe_barrier(PIPE_V);
      RoundMode v505 = RoundMode::CAST_ROUND;
      SaturationMode v506 = SaturationMode::ON;
      TCVT(v503, v499, v505, v506);
      // pto: %t__tmp_v345
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v507 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v345
      ;
      uint64_t v508 = (uint64_t) v54;
      TASSIGN(v507, v508);
      pipe_barrier(PIPE_V);
      TMULS(v507, v503, v96);
      // pto: %ci2_bit_inline1570__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v509 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %ci2_bit_inline1570__ssa_v0
      ;
      uint64_t v510 = (uint64_t) v53;
      TASSIGN(v509, v510);
      pipe_barrier(PIPE_V);
      TSUB(v509, v493, v507);
      // pto: %t__tmp_v346
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v511 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v346
      ;
      uint64_t v512 = (uint64_t) v53;
      TASSIGN(v511, v512);
      pipe_barrier(PIPE_V);
      TMULS(v511, v509, v102);
      // pto: %t__tmp_v347
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v513 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v347
      ;
      uint64_t v514 = (uint64_t) v44;
      TASSIGN(v513, v514);
      pipe_barrier(PIPE_V);
      TSUB(v513, v381, v511);
      // pto: %t__tmp_v348
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v515 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v348
      ;
      uint64_t v516 = (uint64_t) v44;
      TASSIGN(v515, v516);
      pipe_barrier(PIPE_V);
      TADDS(v515, v513, v100);
      // pto: %ci2_inline1593__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v517 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %ci2_inline1593__ssa_v1
      ;
      uint64_t v518 = (uint64_t) v44;
      TASSIGN(v517, v518);
      pipe_barrier(PIPE_V);
      RoundMode v519 = RoundMode::CAST_ROUND;
      SaturationMode v520 = SaturationMode::ON;
      TCVT(v517, v515, v519, v520);
      // pto: %t__tmp_v349
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v521 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v349
      ;
      uint64_t v522 = (uint64_t) v53;
      TASSIGN(v521, v522);
      TGATHER(v521, v371, v415, v379);
      // pto: %max2_inline1531__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v523 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %max2_inline1531__ssa_v1
      ;
      uint64_t v524 = (uint64_t) v53;
      TASSIGN(v523, v524);
      pipe_barrier(PIPE_V);
      TMAX(v523, v371, v521);
      // pto: %t__tmp_v350
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v525 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v350
      ;
      uint64_t v526 = (uint64_t) v54;
      TASSIGN(v525, v526);
      pipe_barrier(PIPE_V);
      TGATHER(v525, v523, v449, v379);
      // pto: %max4_inline1532__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v527 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %max4_inline1532__ssa_v1
      ;
      uint64_t v528 = (uint64_t) v53;
      TASSIGN(v527, v528);
      pipe_barrier(PIPE_V);
      TMAX(v527, v523, v525);
      // pto: %t__tmp_v351
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v529 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v351
      ;
      uint64_t v530 = (uint64_t) v48;
      TASSIGN(v529, v530);
      pipe_barrier(PIPE_V);
      TSUB(v529, v371, v527);
      // pto: %exps_inline1535__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v531 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %exps_inline1535__ssa_v1
      ;
      uint64_t v532 = (uint64_t) v48;
      TASSIGN(v531, v532);
      pipe_barrier(PIPE_V);
      TEXP(v531, v529);
      // pto: %t__tmp_v352
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v533 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v352
      ;
      uint64_t v534 = (uint64_t) v53;
      TASSIGN(v533, v534);
      pipe_barrier(PIPE_V);
      TGATHER(v533, v531, v415, v379);
      // pto: %sum2_inline1542__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v535 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %sum2_inline1542__ssa_v1
      ;
      uint64_t v536 = (uint64_t) v53;
      TASSIGN(v535, v536);
      pipe_barrier(PIPE_V);
      TADD(v535, v531, v533);
      // pto: %t__tmp_v353
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v537 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v353
      ;
      uint64_t v538 = (uint64_t) v54;
      TASSIGN(v537, v538);
      pipe_barrier(PIPE_V);
      TGATHER(v537, v535, v449, v379);
      // pto: %sum4_inline1363__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v539 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %sum4_inline1363__ssa_v1
      ;
      uint64_t v540 = (uint64_t) v53;
      TASSIGN(v539, v540);
      pipe_barrier(PIPE_V);
      TADD(v539, v535, v537);
      // pto: %t__tmp_v354
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v541 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v354
      ;
      uint64_t v542 = (uint64_t) v48;
      TASSIGN(v541, v542);
      pipe_barrier(PIPE_V);
      TDIV(v541, v531, v539);
      // pto: %probs_inline1413__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v543 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %probs_inline1413__ssa_v1
      ;
      uint64_t v544 = (uint64_t) v48;
      TASSIGN(v543, v544);
      pipe_barrier(PIPE_V);
      TADDS(v543, v541, v91);
      // pto: %t__tmp_v355
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v545 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v355
      ;
      uint64_t v546 = (uint64_t) v53;
      TASSIGN(v545, v546);
      pipe_barrier(PIPE_V);
      TGATHER(v545, v543, v483, v379);
      // pto: %col2_inline1546__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v547 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %col2_inline1546__ssa_v4
      ;
      uint64_t v548 = (uint64_t) v53;
      TASSIGN(v547, v548);
      pipe_barrier(PIPE_V);
      TADD(v547, v543, v545);
      // pto: %t__tmp_v356
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v549 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v356
      ;
      uint64_t v550 = (uint64_t) v54;
      TASSIGN(v549, v550);
      pipe_barrier(PIPE_V);
      TGATHER(v549, v547, v517, v379);
      // pto: %t__tmp_v357
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v551 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %t__tmp_v357
      ;
      uint64_t v552 = (uint64_t) v53;
      TASSIGN(v551, v552);
      pipe_barrier(PIPE_V);
      TADD(v551, v547, v549);
      // pto: %col4_inline1495__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v553 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %col4_inline1495__ssa_v4
      ;
      uint64_t v554 = (uint64_t) v53;
      TASSIGN(v553, v554);
      pipe_barrier(PIPE_V);
      TADDS(v553, v551, v91);
      // pto: %cur_inline1530__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v555 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
      // pto: %cur_inline1530__ssa_v4
      ;
      uint64_t v556 = (uint64_t) v48;
      TASSIGN(v555, v556);
      pipe_barrier(PIPE_V);
      TDIV(v555, v543, v553);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      for (int64_t v557 = v85; v557 < v103; v557 += v74) {
        // pto: %t__tmp_v358
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v558 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
        // pto: %t__tmp_v358
        ;
        uint64_t v559 = (uint64_t) v53;
        TASSIGN(v558, v559);
        pipe_barrier(PIPE_V);
        TGATHER(v558, v555, v415, v379);
        // pto: %row2_inline1378__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v560 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
        // pto: %row2_inline1378__ssa_v1
        ;
        uint64_t v561 = (uint64_t) v53;
        TASSIGN(v560, v561);
        pipe_barrier(PIPE_V);
        TADD(v560, v555, v558);
        // pto: %t__tmp_v359
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v562 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
        // pto: %t__tmp_v359
        ;
        uint64_t v563 = (uint64_t) v54;
        TASSIGN(v562, v563);
        pipe_barrier(PIPE_V);
        TGATHER(v562, v560, v449, v379);
        // pto: %t__tmp_v360
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v564 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
        // pto: %t__tmp_v360
        ;
        uint64_t v565 = (uint64_t) v53;
        TASSIGN(v564, v565);
        pipe_barrier(PIPE_V);
        TADD(v564, v560, v562);
        // pto: %row4_inline1344__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v566 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
        // pto: %row4_inline1344__ssa_v1
        ;
        uint64_t v567 = (uint64_t) v53;
        TASSIGN(v566, v567);
        pipe_barrier(PIPE_V);
        TADDS(v566, v564, v91);
        // pto: %norm_inline1499__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v568 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
        // pto: %norm_inline1499__ssa_v1
        ;
        uint64_t v569 = (uint64_t) v53;
        TASSIGN(v568, v569);
        pipe_barrier(PIPE_V);
        TDIV(v568, v555, v566);
        // pto: %t__tmp_v361
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v570 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
        // pto: %t__tmp_v361
        ;
        uint64_t v571 = (uint64_t) v54;
        TASSIGN(v570, v571);
        pipe_barrier(PIPE_V);
        TGATHER(v570, v568, v483, v379);
        // pto: %col2_inline1546__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v572 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
        // pto: %col2_inline1546__ssa_v7
        ;
        uint64_t v573 = (uint64_t) v54;
        TASSIGN(v572, v573);
        pipe_barrier(PIPE_V);
        TADD(v572, v568, v570);
        // pto: %t__tmp_v362
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v574 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
        // pto: %t__tmp_v362
        ;
        uint64_t v575 = (uint64_t) v55;
        TASSIGN(v574, v575);
        pipe_barrier(PIPE_V);
        TGATHER(v574, v572, v517, v379);
        // pto: %t__tmp_v363
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v576 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
        // pto: %t__tmp_v363
        ;
        uint64_t v577 = (uint64_t) v54;
        TASSIGN(v576, v577);
        pipe_barrier(PIPE_V);
        TADD(v576, v572, v574);
        // pto: %col4_inline1495__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v578 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
        // pto: %col4_inline1495__ssa_v7
        ;
        uint64_t v579 = (uint64_t) v54;
        TASSIGN(v578, v579);
        pipe_barrier(PIPE_V);
        TADDS(v578, v576, v91);
        // pto: %cur_inline1530__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v580 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v92);
        // pto: %cur_inline1530__ssa_v7
        ;
        uint64_t v581 = (uint64_t) v48;
        TASSIGN(v580, v581);
        pipe_barrier(PIPE_V);
        TDIV(v580, v568, v578);
      };
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      // pto: %t__tmp_v364
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v582 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v79);
      // pto: %t__tmp_v364
      ;
      uint64_t v583 = (uint64_t) v48;
      TASSIGN(v582, v583);
      v582.SetValidShape(v286, v79);
      // pto: %comb_inline1434__ssa_v0_pview
      ;
      const int64_t v584 = 0;
      // pto: %comb_inline1434__ssa_v0_pview
      ;
      __gm__ float* v585 = PTOAS__GLOBAL_TENSOR_DATA(v159);
      // pto: %comb_inline1434__ssa_v0_pview
      ;
      const int64_t v586 = 1;
      // pto: %comb_inline1434__ssa_v0_pview
      ;
      const int64_t v587 = 1;
      // pto: %comb_inline1434__ssa_v0_pview
      ;
      const int64_t v588 = 1;
      // pto: %comb_inline1434__ssa_v0_pview
      ;
      int64_t v589 = v286 * v79;
      // pto: %comb_inline1434__ssa_v0_pview
      ;
      int64_t v590 = v588 * v589;
      // pto: %comb_inline1434__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, 16> v591 = pto::Shape<1, 1, 1, -1, 16>(v586, v587, v588, v286, v79);
      // pto: %comb_inline1434__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v592 = pto::Stride<-1, -1, -1, -1, -1>(v587 * v590, v590, v589, v79, v74);
      // pto: %comb_inline1434__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v593 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v585 + (v584 + v316 * v79 + v85 * v74), v591, v592);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      TSTORE(v593, v582);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    };
  } else {
    // pto: %101
    ;
    int64_t v594 = (int64_t) ((uint64_t) v276 - (uint64_t) v18);
    // pto: %103
    ;
    int64_t v595 = v594 < v22 ? v594 : v22;
    // pto: %102, %104, %105
    ;
    int64_t v596 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v594 * (uint64_t) v21) + (uint64_t) v595) * (uint64_t) v87);
    // pto: %106
    ;
    int64_t v597 = (int64_t) ((uint64_t) v22 - (uint64_t) v594);
    // pto: %107
    ;
    int64_t v598 = v597 < v85 ? v85 : v597;
    // pto: %108
    ;
    int64_t v599 = v598 < v74 ? v598 : v74;
    // pto: %111, %109, %110
    ;
    int64_t v600 = (int64_t) ((uint64_t) v596 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v21 + (uint64_t) v599) * (uint64_t) v87));
    // pto: %112
    ;
    // pto: %113
    ;
    // pto: %d_begin_inline1468__phi_v2
    ;
    int64_t v601;
    // pto: %d_end_inline1589__phi_v2
    ;
    int64_t v602;
    // pto: %work_begin_inline1346__phi_v3
    ;
    int64_t v603;
    // pto: %work_end_inline1422__phi_v3
    ;
    int64_t v604;
    if (v24 < v25) {
      // pto: %114
      ;
      // pto: %d_worker_inline1597__phi_v2
      ;
      int64_t v605;
      // pto: %d_workers_inline1585__phi_v2
      ;
      int64_t v606;
      // pto: %pair_id_inline1594__phi_v2
      ;
      int64_t v607;
      if (v594 < v26) {
        // pto: %115
        ;
        int64_t v608 = (int64_t) ((uint64_t) v27 + (uint64_t) v74);
        // pto: %116
        ;
        // pto: %118
        ;
        v605 = v594 % v608;
        v606 = v608;
        v607 = v594 / v608;
      } else {
        // pto: %120
        ;
        int64_t v609 = (int64_t) ((uint64_t) v594 - (uint64_t) v26);
        // pto: %122, %121
        ;
        // pto: %123
        ;
        v605 = v609 % v27;
        v606 = v27;
        v607 = (int64_t) ((uint64_t) v28 + (uint64_t) (v609 / v27));
      };
      // pto: %124
      ;
      uint64_t v610 = (uint64_t) v607;
      // pto: %124
      ;
      int64_t v611 = (int64_t) (v610 * (uint64_t) v87);
      // pto: %125
      ;
      int64_t v612 = v23 < v611 ? v23 : v611;
      // pto: %126
      ;
      int64_t v613 = (int64_t) ((uint64_t) v612 + (uint64_t) v87);
      // pto: %127
      ;
      // pto: %128
      ;
      int64_t v614 = v78 / v606;
      // pto: %129
      ;
      int64_t v615 = v78 % v606;
      // pto: %130
      ;
      uint64_t v616 = (uint64_t) v605;
      // pto: %131
      ;
      bool v617 = v605 < v615;
      // pto: %131
      ;
      int64_t v618 = v617 ? v605 : v615;
      // pto: %130, %132
      ;
      int64_t v619 = (int64_t) ((uint64_t) ((int64_t) v616 * (uint64_t) v614) + (uint64_t) v618);
      // pto: %134
      ;
      uint64_t v620 = (uint64_t) v605;
      // pto: %134
      ;
      int64_t v621 = (int64_t) ((uint64_t) v615 - v620);
      // pto: %135
      ;
      int64_t v622 = v621 < v85 ? v85 : v621;
      // pto: %136
      ;
      int64_t v623 = v622 < v74 ? v622 : v74;
      // pto: %133, %137
      ;
      v601 = v619;
      v602 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v619 + (uint64_t) v614) + (uint64_t) v623);
      v603 = v612;
      v604 = v23 < v613 ? v23 : v613;
    } else {
      v601 = v85;
      v602 = v78;
      v603 = v596;
      v604 = v23 < v600 ? v23 : v600;
    };
    // pto: %138
    ;
    bool v624 = v25 <= v594;
    // pto: %work_begin_inline1346__phi_v5
    ;
    int64_t v625 = v624 ? v85 : v603;
    // pto: %work_end_inline1422__phi_v5
    ;
    int64_t v626 = v624 ? v85 : v604;
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    for (int64_t v627 = v625; v627 < v626; v627 += v78) {
      // pto: %139
      ;
      int64_t v628 = (int64_t) ((uint64_t) v626 - (uint64_t) v627);
      // pto: %140
      ;
      int64_t v629 = v628 < v78 ? v628 : v78;
      // pto: %141
      ;
      int64_t v630 = (int64_t) ((uint64_t) v629 * (uint64_t) v73);
      // pto: %m_parts_load_inline1599__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v631 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v630);
      // pto: %m_parts_load_inline1599__ssa_v0
      ;
      uint64_t v632 = (uint64_t) v40;
      TASSIGN(v631, v632);
      // pto: %142
      ;
      int64_t v633 = (int64_t) ((uint64_t) v627 * (uint64_t) v73);
      // pto: %143
      ;
      int64_t v634 = v633 < v85 ? v85 : v633;
      // pto: %145
      ;
      const int64_t v635 = 0;
      // pto: %145
      ;
      __gm__ float* v636 = PTOAS__GLOBAL_TENSOR_DATA(v133);
      // pto: %145
      ;
      const int64_t v637 = 1;
      // pto: %145
      ;
      const int64_t v638 = 1;
      // pto: %145
      ;
      const int64_t v639 = 1;
      // pto: %145
      ;
      int64_t v640 = v78 * v125;
      // pto: %145
      ;
      int64_t v641 = v639 * v640;
      // pto: %145
      ;
      pto::Shape<1, 1, 1, 8, -1> v642 = pto::Shape<1, 1, 1, 8, -1>(v637, v638, v639, v78, v630);
      // pto: %145
      ;
      pto::Stride<-1, -1, -1, -1, -1> v643 = pto::Stride<-1, -1, -1, -1, -1>(v638 * v641, v641, v640, v125, v74);
      // pto: %145
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v644 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v636 + (v635 + v85 * v125 + v634 * v74), v642, v643);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      TLOAD(v631, v644);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      // pto: %t__tmp_v365
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v645 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v78, v89);
      // pto: %t__tmp_v365
      ;
      uint64_t v646 = (uint64_t) v40;
      TASSIGN(v645, v646);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      TFILLPAD<pto::TFillPadMode::InPlace>(v645, v631);
      v645.SetValidShape(v78, v89);
      // pto: %m_parts_tmp_inline1601__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v647 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v89);
      // pto: %m_parts_tmp_inline1601__ssa_v0
      ;
      uint64_t v648 = (uint64_t) v42;
      TASSIGN(v647, v648);
      // pto: %t__tmp_v366
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v649 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v89);
      // pto: %t__tmp_v366
      ;
      uint64_t v650 = (uint64_t) v40;
      TASSIGN(v649, v650);
      // pto: %146
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, 8, 256, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v651;
      TRESHAPE(v651, v645);
      // pto: %147
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, 8, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v652;
      TRESHAPE(v652, v647);
      // pto: %148
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v653;
      TRESHAPE(v653, v649);
      pipe_barrier(PIPE_V);
      const bool v654 = true;
      TCOLSUM(v653, v651, v652, v654);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      // pto: %m_total_inline1357__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v655 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v73);
      // pto: %m_total_inline1357__ssa_v0
      ;
      uint64_t v656 = (uint64_t) v40;
      TASSIGN(v655, v656);
      // pto: %m_sq_load_inline1596__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v657 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v629);
      // pto: %m_sq_load_inline1596__ssa_v0
      ;
      uint64_t v658 = (uint64_t) v43;
      TASSIGN(v657, v658);
      // pto: %149
      ;
      int64_t v659 = v627 < v85 ? v85 : v627;
      // pto: %150
      ;
      int64_t v660 = (int64_t) v29;
      // pto: %150
      ;
      const int64_t v661 = 0;
      // pto: %150
      ;
      __gm__ float* v662 = PTOAS__GLOBAL_TENSOR_DATA(v142);
      // pto: %150
      ;
      const int64_t v663 = 1;
      // pto: %150
      ;
      const int64_t v664 = 1;
      // pto: %150
      ;
      const int64_t v665 = 1;
      // pto: %150
      ;
      int64_t v666 = v78 * v660;
      // pto: %150
      ;
      int64_t v667 = v665 * v666;
      // pto: %150
      ;
      pto::Shape<1, 1, 1, 8, -1> v668 = pto::Shape<1, 1, 1, 8, -1>(v663, v664, v665, v78, v629);
      // pto: %150
      ;
      pto::Stride<-1, -1, -1, -1, -1> v669 = pto::Stride<-1, -1, -1, -1, -1>(v664 * v667, v667, v666, v660, v74);
      // pto: %150
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v670 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v662 + (v661 + v85 * v660 + v659 * v74), v668, v669);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      TLOAD(v657, v670);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      // pto: %t__tmp_v367
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v671 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v78, v78);
      // pto: %t__tmp_v367
      ;
      uint64_t v672 = (uint64_t) v43;
      TASSIGN(v671, v672);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      TFILLPAD<pto::TFillPadMode::InPlace>(v671, v657);
      v671.SetValidShape(v78, v78);
      // pto: %m_sq_tmp_inline1487__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v673 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %m_sq_tmp_inline1487__ssa_v0
      ;
      uint64_t v674 = (uint64_t) v44;
      TASSIGN(v673, v674);
      // pto: %m_rms_tmp_inline1605__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v675 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %m_rms_tmp_inline1605__ssa_v0
      ;
      uint64_t v676 = (uint64_t) v45;
      TASSIGN(v675, v676);
      // pto: %t__tmp_v368
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v677 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %t__tmp_v368
      ;
      uint64_t v678 = (uint64_t) v43;
      TASSIGN(v677, v678);
      // pto: %151
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v679;
      TRESHAPE(v679, v671);
      // pto: %152
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v680;
      TRESHAPE(v680, v673);
      // pto: %153
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v681;
      TRESHAPE(v681, v677);
      pipe_barrier(PIPE_V);
      const bool v682 = true;
      TCOLSUM(v681, v679, v680, v682);
      // pto: %t__tmp_v369
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v683 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %t__tmp_v369
      ;
      uint64_t v684 = (uint64_t) v43;
      TASSIGN(v683, v684);
      pipe_barrier(PIPE_V);
      TMULS(v683, v677, v90);
      // pto: %t__tmp_v370
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v685 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %t__tmp_v370
      ;
      uint64_t v686 = (uint64_t) v43;
      TASSIGN(v685, v686);
      pipe_barrier(PIPE_V);
      TADDS(v685, v683, v91);
      // pto: %t__tmp_v371
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v687 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %t__tmp_v371
      ;
      uint64_t v688 = (uint64_t) v50;
      TASSIGN(v687, v688);
      pipe_barrier(PIPE_V);
      TRSQRT(v687, v685, v675);
      // pto: %m_inv_inline1606__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v689 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v74);
      // pto: %m_inv_inline1606__ssa_v0
      ;
      uint64_t v690 = (uint64_t) v50;
      TASSIGN(v689, v690);
      // pto: %gate_base_inline1592__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v691 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %gate_base_inline1592__ssa_v0
      ;
      uint64_t v692 = (uint64_t) v56;
      TASSIGN(v691, v692);
      // pto: %154
      ;
      __gm__ float* v693 = PTOAS__GLOBAL_TENSOR_DATA(v150);
      // pto: %154
      ;
      pto::Shape<1, 1, 1, 1, 8> v694 = pto::Shape<1, 1, 1, 1, 8>();
      // pto: %154
      ;
      pto::Stride<24, 24, 24, 24, 1> v695 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %154
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v696 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v693, v694, v695);
      TLOAD(v691, v696);
      // pto: %gate_scales_inline1538__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v697 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v77);
      // pto: %gate_scales_inline1538__ssa_v0
      ;
      uint64_t v698 = (uint64_t) v57;
      TASSIGN(v697, v698);
      // pto: %hc_scale_2d_inline1448__ssa_v0_pview
      ;
      __gm__ float* v699 = PTOAS__GLOBAL_TENSOR_DATA(v167);
      // pto: %hc_scale_2d_inline1448__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 3> v700 = pto::Shape<1, 1, 1, 1, 3>();
      // pto: %hc_scale_2d_inline1448__ssa_v0_pview
      ;
      pto::Stride<3, 3, 3, 3, 1> v701 = pto::Stride<3, 3, 3, 3, 1>();
      // pto: %hc_scale_2d_inline1448__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND> v702 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND>(v699, v700, v701);
      TLOAD(v697, v702);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
      // pto: %t__ci_tmp_v1
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v703 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v93);
      // pto: %t__ci_tmp_v1
      ;
      uint64_t v704 = (uint64_t) v43;
      TASSIGN(v703, v704);
      // pto: %t__tmp_v372
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v705 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %t__tmp_v372
      ;
      uint64_t v706 = (uint64_t) v58;
      TASSIGN(v705, v706);
      // pto: %155
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v707;
      TRESHAPE(v707, v703);
      // pto: %156
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v708;
      TRESHAPE(v708, v705);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v708, v81, v707);
      // pto: %gate_ids_inline1607__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v709 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %gate_ids_inline1607__ssa_v0
      ;
      uint64_t v710 = (uint64_t) v43;
      TASSIGN(v709, v710);
      pipe_barrier(PIPE_V);
      RoundMode v711 = RoundMode::CAST_ROUND;
      SaturationMode v712 = SaturationMode::ON;
      TCVT(v709, v705, v711, v712);
      // pto: %t__tmp_v373
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v713 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %t__tmp_v373
      ;
      uint64_t v714 = (uint64_t) v43;
      TASSIGN(v713, v714);
      pipe_barrier(PIPE_V);
      TMULS(v713, v709, v97);
      // pto: %gate_scale_ids_inline1483__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v715 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %gate_scale_ids_inline1483__ssa_v0
      ;
      uint64_t v716 = (uint64_t) v43;
      TASSIGN(v715, v716);
      pipe_barrier(PIPE_V);
      RoundMode v717 = RoundMode::CAST_FLOOR;
      SaturationMode v718 = SaturationMode::ON;
      TCVT(v715, v713, v717, v718);
      // pto: %gate_tmp_inline1608__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v719 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %gate_tmp_inline1608__ssa_v0
      ;
      uint64_t v720 = (uint64_t) v46;
      TASSIGN(v719, v720);
      // pto: %gate_scale_vec_inline1521__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v721 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %gate_scale_vec_inline1521__ssa_v0
      ;
      uint64_t v722 = (uint64_t) v59;
      TASSIGN(v721, v722);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
      TGATHER(v721, v697, v715, v719);
      // pto: %t__tmp_v374
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v723 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %t__tmp_v374
      ;
      uint64_t v724 = (uint64_t) v40;
      TASSIGN(v723, v724);
      // pto: %157
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v725;
      // pto: %157
      ;
      uint64_t v726 = (uint64_t) v40;
      TASSIGN(v725, v726);
      // pto: %gate_raw_inline1609__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v727 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %gate_raw_inline1609__ssa_v0
      ;
      uint64_t v728 = (uint64_t) v40;
      TASSIGN(v727, v728);
      TADDS(v727, v725, v84);
      // pto: %gate_norm_inline1339__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v729 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %gate_norm_inline1339__ssa_v0
      ;
      uint64_t v730 = (uint64_t) v40;
      TASSIGN(v729, v730);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v729, v727, v689);
      // pto: %gate_scaled_inline1373__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v731 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %gate_scaled_inline1373__ssa_v0
      ;
      uint64_t v732 = (uint64_t) v40;
      TASSIGN(v731, v732);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v731, v729, v721);
      // pto: %t__tmp_v375
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v733 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %t__tmp_v375
      ;
      uint64_t v734 = (uint64_t) v60;
      TASSIGN(v733, v734);
      TCOLEXPAND(v733, v691);
      // pto: %gate_logits_inline1489__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v735 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %gate_logits_inline1489__ssa_v0
      ;
      uint64_t v736 = (uint64_t) v40;
      TASSIGN(v735, v736);
      pipe_barrier(PIPE_V);
      TADD(v735, v731, v733);
      // pto: %t__tmp_v376
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v737 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %t__tmp_v376
      ;
      uint64_t v738 = (uint64_t) v40;
      TASSIGN(v737, v738);
      pipe_barrier(PIPE_V);
      TNEG(v737, v735);
      // pto: %t__tmp_v377
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v739 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %t__tmp_v377
      ;
      uint64_t v740 = (uint64_t) v40;
      TASSIGN(v739, v740);
      pipe_barrier(PIPE_V);
      TEXP(v739, v737);
      // pto: %t__tmp_v378
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v741 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %t__tmp_v378
      ;
      uint64_t v742 = (uint64_t) v40;
      TASSIGN(v741, v742);
      pipe_barrier(PIPE_V);
      TADDS(v741, v739, v94);
      // pto: %gate_sigmoid_inline1334__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v743 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %gate_sigmoid_inline1334__ssa_v0
      ;
      uint64_t v744 = (uint64_t) v49;
      TASSIGN(v743, v744);
      pipe_barrier(PIPE_V);
      TRECIP(v743, v741);
      // pto: %transpose_tmp
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v745 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %transpose_tmp
      ;
      uint64_t v746 = (uint64_t) v40;
      TASSIGN(v745, v746);
      // pto: %gate_transposed_inline1449__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v747 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %gate_transposed_inline1449__ssa_v0
      ;
      uint64_t v748 = (uint64_t) v43;
      TASSIGN(v747, v748);
      pipe_barrier(PIPE_V);
      TTRANS(v747, v743, v745);
      // pto: %t__ci_tmp_v2
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v749 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v93);
      // pto: %t__ci_tmp_v2
      ;
      uint64_t v750 = (uint64_t) v40;
      TASSIGN(v749, v750);
      // pto: %t__tmp_v379
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v751 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v73);
      // pto: %t__tmp_v379
      ;
      uint64_t v752 = (uint64_t) v44;
      TASSIGN(v751, v752);
      // pto: %158
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v753;
      TRESHAPE(v753, v749);
      // pto: %159
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v754;
      TRESHAPE(v754, v751);
      TCI<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v754, v81, v753);
      // pto: %post_ids_float_inline1333__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v755 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v73);
      // pto: %post_ids_float_inline1333__ssa_v0
      ;
      uint64_t v756 = (uint64_t) v40;
      TASSIGN(v755, v756);
      pipe_barrier(PIPE_V);
      RoundMode v757 = RoundMode::CAST_ROUND;
      SaturationMode v758 = SaturationMode::ON;
      TCVT(v755, v751, v757, v758);
      // pto: %t__tmp_v380
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v759 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v73);
      // pto: %t__tmp_v380
      ;
      uint64_t v760 = (uint64_t) v61;
      TASSIGN(v759, v760);
      pipe_barrier(PIPE_V);
      TMULS(v759, v755, v97);
      // pto: %t__tmp_v381
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v761 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v73);
      // pto: %t__tmp_v381
      ;
      uint64_t v762 = (uint64_t) v61;
      TASSIGN(v761, v762);
      pipe_barrier(PIPE_V);
      RoundMode v763 = RoundMode::CAST_FLOOR;
      SaturationMode v764 = SaturationMode::ON;
      TCVT(v761, v759, v763, v764);
      // pto: %post_rows_inline1423__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v765 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v73);
      // pto: %post_rows_inline1423__ssa_v0
      ;
      uint64_t v766 = (uint64_t) v61;
      TASSIGN(v765, v766);
      pipe_barrier(PIPE_V);
      RoundMode v767 = RoundMode::CAST_ROUND;
      SaturationMode v768 = SaturationMode::ON;
      TCVT(v765, v761, v767, v768);
      // pto: %t__tmp_v382
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v769 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v73);
      // pto: %t__tmp_v382
      ;
      uint64_t v770 = (uint64_t) v61;
      TASSIGN(v769, v770);
      pipe_barrier(PIPE_V);
      TMULS(v769, v765, v98);
      // pto: %t__tmp_v383
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v771 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v73);
      // pto: %t__tmp_v383
      ;
      uint64_t v772 = (uint64_t) v40;
      TASSIGN(v771, v772);
      pipe_barrier(PIPE_V);
      TADD(v771, v755, v769);
      // pto: %t__tmp_v384
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v773 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v73);
      // pto: %t__tmp_v384
      ;
      uint64_t v774 = (uint64_t) v40;
      TASSIGN(v773, v774);
      pipe_barrier(PIPE_V);
      TADDS(v773, v771, v98);
      // pto: %post_ids_inline1395__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v775 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v73);
      // pto: %post_ids_inline1395__ssa_v0
      ;
      uint64_t v776 = (uint64_t) v40;
      TASSIGN(v775, v776);
      pipe_barrier(PIPE_V);
      RoundMode v777 = RoundMode::CAST_ROUND;
      SaturationMode v778 = SaturationMode::ON;
      TCVT(v775, v773, v777, v778);
      // pto: %post_tmp_inline1332__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v779 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v73);
      // pto: %post_tmp_inline1332__ssa_v0
      ;
      uint64_t v780 = (uint64_t) v61;
      TASSIGN(v779, v780);
      // pto: %t__tmp_v385
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v781 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v104);
      // pto: %t__tmp_v385
      ;
      uint64_t v782 = (uint64_t) v49;
      TASSIGN(v781, v782);
      // pto: %post_values_inline1331__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v783 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v73);
      // pto: %post_values_inline1331__ssa_v0
      ;
      uint64_t v784 = (uint64_t) v60;
      TASSIGN(v783, v784);
      pipe_barrier(PIPE_V);
      TGATHER(v783, v781, v775, v779);
      // pto: %t__tmp_v386
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v785 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v73);
      // pto: %t__tmp_v386
      ;
      uint64_t v786 = (uint64_t) v40;
      TASSIGN(v785, v786);
      pipe_barrier(PIPE_V);
      TMULS(v785, v783, v96);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
      // pto: %160
      ;
      int64_t v787 = (int64_t) ((uint64_t) v629 * (uint64_t) v86);
      v785.SetValidShape(v74, v787);
      // pto: %161
      ;
      bool v788 = v601 == v85;
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
      if (v788) {
        // pto: %163
        ;
        int64_t v789 = (int64_t) ((uint64_t) v627 * (uint64_t) v86);
        // pto: %164
        ;
        int64_t v790 = v789 < v85 ? v85 : v789;
        // pto: %post_flat_inline1440__ssa_v0_pview
        ;
        const int64_t v791 = 0;
        // pto: %post_flat_inline1440__ssa_v0_pview
        ;
        __gm__ float* v792 = PTOAS__GLOBAL_TENSOR_DATA(v176);
        // pto: %post_flat_inline1440__ssa_v0_pview
        ;
        const int64_t v793 = 1;
        // pto: %post_flat_inline1440__ssa_v0_pview
        ;
        const int64_t v794 = 1;
        // pto: %post_flat_inline1440__ssa_v0_pview
        ;
        const int64_t v795 = 1;
        // pto: %post_flat_inline1440__ssa_v0_pview
        ;
        int64_t v796 = v74 * v168;
        // pto: %post_flat_inline1440__ssa_v0_pview
        ;
        int64_t v797 = v795 * v796;
        // pto: %post_flat_inline1440__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, -1> v798 = pto::Shape<1, 1, 1, 1, -1>(v793, v794, v795, v74, v787);
        // pto: %post_flat_inline1440__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v799 = pto::Stride<-1, -1, -1, -1, -1>(v794 * v797, v797, v796, v168, v74);
        // pto: %post_flat_inline1440__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v800 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v792 + (v791 + v85 * v168 + v790 * v74), v798, v799);
        TSTORE(v800, v785);
      };
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      // pto: %t__tmp_v387
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v801 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %t__tmp_v387
      ;
      uint64_t v802 = (uint64_t) v43;
      TASSIGN(v801, v802);
      // pto: %165
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v803;
      // pto: %165
      ;
      uint64_t v804 = (uint64_t) v43;
      TASSIGN(v803, v804);
      // pto: %t__tmp_v388
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v805 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %t__tmp_v388
      ;
      uint64_t v806 = (uint64_t) v62;
      TASSIGN(v805, v806);
      TADDS(v805, v803, v91);
      // pto: %pre0_inline1297__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v807 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v74);
      // pto: %pre0_inline1297__ssa_v0
      ;
      uint64_t v808 = (uint64_t) v62;
      TASSIGN(v807, v808);
      // pto: %t__tmp_v389
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v809 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %t__tmp_v389
      ;
      uint64_t v810 = (uint64_t) v46;
      TASSIGN(v809, v810);
      // pto: %166
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v811;
      // pto: %166
      ;
      uint64_t v812 = (uint64_t) v46;
      TASSIGN(v811, v812);
      // pto: %t__tmp_v390
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v813 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %t__tmp_v390
      ;
      uint64_t v814 = (uint64_t) v63;
      TASSIGN(v813, v814);
      TADDS(v813, v811, v91);
      // pto: %pre1_inline1345__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v815 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v74);
      // pto: %pre1_inline1345__ssa_v0
      ;
      uint64_t v816 = (uint64_t) v63;
      TASSIGN(v815, v816);
      // pto: %t__tmp_v391
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v817 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %t__tmp_v391
      ;
      uint64_t v818 = (uint64_t) v59;
      TASSIGN(v817, v818);
      // pto: %167
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v819;
      // pto: %167
      ;
      uint64_t v820 = (uint64_t) v59;
      TASSIGN(v819, v820);
      // pto: %t__tmp_v392
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v821 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %t__tmp_v392
      ;
      uint64_t v822 = (uint64_t) v64;
      TASSIGN(v821, v822);
      TADDS(v821, v819, v91);
      // pto: %pre2_inline1451__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v823 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v74);
      // pto: %pre2_inline1451__ssa_v0
      ;
      uint64_t v824 = (uint64_t) v64;
      TASSIGN(v823, v824);
      // pto: %t__tmp_v393
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v825 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %t__tmp_v393
      ;
      uint64_t v826 = (uint64_t) v65;
      TASSIGN(v825, v826);
      // pto: %168
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v827;
      // pto: %168
      ;
      uint64_t v828 = (uint64_t) v65;
      TASSIGN(v827, v828);
      // pto: %t__tmp_v394
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v829 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v78);
      // pto: %t__tmp_v394
      ;
      uint64_t v830 = (uint64_t) v66;
      TASSIGN(v829, v830);
      TADDS(v829, v827, v91);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      // pto: %pre3_inline1296__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v831 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v74);
      // pto: %pre3_inline1296__ssa_v0
      ;
      uint64_t v832 = (uint64_t) v66;
      TASSIGN(v831, v832);
      // pto: %169
      ;
      uint64_t v833 = (uint64_t) v602;
      // pto: %169
      ;
      uint64_t v834 = (uint64_t) v601;
      // pto: %172
      ;
      uint64_t v835 = (uint64_t) v601;
      // pto: %169, %170, %171, %172
      ;
      int64_t v836 = (int64_t) (v835 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) (v833 - v834) / v87) * (uint64_t) v87));
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      for (int64_t v837 = v601; v837 < v836; v837 += v87) {
        // pto: %173
        ;
        int64_t v838 = (int64_t) ((uint64_t) v837 * (uint64_t) v88);
        // pto: %176
        ;
        int64_t v839 = (int64_t) ((uint64_t) v838 + (uint64_t) v88);
        // pto: %x0_bf16_inline1565__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v840 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %x0_bf16_inline1565__ssa_v0
        ;
        uint64_t v841 = (uint64_t) v67;
        TASSIGN(v840, v841);
        // pto: %179
        ;
        int64_t v842 = v838 < v85 ? v85 : v838;
        // pto: %180
        ;
        const int64_t v843 = 0;
        // pto: %180
        ;
        __gm__ bfloat16_t* v844 = PTOAS__GLOBAL_TENSOR_DATA(v124);
        // pto: %180
        ;
        const int64_t v845 = 1;
        // pto: %180
        ;
        const int64_t v846 = 1;
        // pto: %180
        ;
        const int64_t v847 = 1;
        // pto: %180
        ;
        int64_t v848 = v629 * v76;
        // pto: %180
        ;
        int64_t v849 = v847 * v848;
        // pto: %180
        ;
        pto::Shape<1, 1, 1, -1, 512> v850 = pto::Shape<1, 1, 1, -1, 512>(v845, v846, v847, v629, v88);
        // pto: %180
        ;
        pto::Stride<-1, -1, -1, -1, -1> v851 = pto::Stride<-1, -1, -1, -1, -1>(v846 * v849, v849, v848, v76, v74);
        // pto: %180
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v852 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v844 + (v843 + v659 * v76 + v842 * v74), v850, v851);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v840, v852);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %x1_bf16_inline1329__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v853 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %x1_bf16_inline1329__ssa_v0
        ;
        uint64_t v854 = (uint64_t) v68;
        TASSIGN(v853, v854);
        // pto: %182
        ;
        int64_t v855 = (int64_t) ((uint64_t) v838 + (uint64_t) v80);
        // pto: %183
        ;
        int64_t v856 = v855 < v85 ? v85 : v855;
        // pto: %184
        ;
        const int64_t v857 = 0;
        // pto: %184
        ;
        __gm__ bfloat16_t* v858 = PTOAS__GLOBAL_TENSOR_DATA(v124);
        // pto: %184
        ;
        const int64_t v859 = 1;
        // pto: %184
        ;
        const int64_t v860 = 1;
        // pto: %184
        ;
        const int64_t v861 = 1;
        // pto: %184
        ;
        int64_t v862 = v629 * v76;
        // pto: %184
        ;
        int64_t v863 = v861 * v862;
        // pto: %184
        ;
        pto::Shape<1, 1, 1, -1, 512> v864 = pto::Shape<1, 1, 1, -1, 512>(v859, v860, v861, v629, v88);
        // pto: %184
        ;
        pto::Stride<-1, -1, -1, -1, -1> v865 = pto::Stride<-1, -1, -1, -1, -1>(v860 * v863, v863, v862, v76, v74);
        // pto: %184
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v866 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v858 + (v857 + v659 * v76 + v856 * v74), v864, v865);
        TLOAD(v853, v866);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %x2_bf16_inline1292__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v867 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %x2_bf16_inline1292__ssa_v0
        ;
        uint64_t v868 = (uint64_t) v38;
        TASSIGN(v867, v868);
        // pto: %186
        ;
        int64_t v869 = (int64_t) ((uint64_t) v838 + (uint64_t) v105);
        // pto: %187
        ;
        int64_t v870 = v869 < v85 ? v85 : v869;
        // pto: %188
        ;
        const int64_t v871 = 0;
        // pto: %188
        ;
        __gm__ bfloat16_t* v872 = PTOAS__GLOBAL_TENSOR_DATA(v124);
        // pto: %188
        ;
        const int64_t v873 = 1;
        // pto: %188
        ;
        const int64_t v874 = 1;
        // pto: %188
        ;
        const int64_t v875 = 1;
        // pto: %188
        ;
        int64_t v876 = v629 * v76;
        // pto: %188
        ;
        int64_t v877 = v875 * v876;
        // pto: %188
        ;
        pto::Shape<1, 1, 1, -1, 512> v878 = pto::Shape<1, 1, 1, -1, 512>(v873, v874, v875, v629, v88);
        // pto: %188
        ;
        pto::Stride<-1, -1, -1, -1, -1> v879 = pto::Stride<-1, -1, -1, -1, -1>(v874 * v877, v877, v876, v76, v74);
        // pto: %188
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v880 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v872 + (v871 + v659 * v76 + v870 * v74), v878, v879);
        TLOAD(v867, v880);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %x3_bf16_inline1356__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v881 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %x3_bf16_inline1356__ssa_v0
        ;
        uint64_t v882 = (uint64_t) v69;
        TASSIGN(v881, v882);
        // pto: %190
        ;
        int64_t v883 = (int64_t) ((uint64_t) v838 + (uint64_t) v106);
        // pto: %191
        ;
        int64_t v884 = v883 < v85 ? v85 : v883;
        // pto: %192
        ;
        const int64_t v885 = 0;
        // pto: %192
        ;
        __gm__ bfloat16_t* v886 = PTOAS__GLOBAL_TENSOR_DATA(v124);
        // pto: %192
        ;
        const int64_t v887 = 1;
        // pto: %192
        ;
        const int64_t v888 = 1;
        // pto: %192
        ;
        const int64_t v889 = 1;
        // pto: %192
        ;
        int64_t v890 = v629 * v76;
        // pto: %192
        ;
        int64_t v891 = v889 * v890;
        // pto: %192
        ;
        pto::Shape<1, 1, 1, -1, 512> v892 = pto::Shape<1, 1, 1, -1, 512>(v887, v888, v889, v629, v88);
        // pto: %192
        ;
        pto::Stride<-1, -1, -1, -1, -1> v893 = pto::Stride<-1, -1, -1, -1, -1>(v888 * v891, v891, v890, v76, v74);
        // pto: %192
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v894 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v886 + (v885 + v659 * v76 + v884 * v74), v892, v893);
        TLOAD(v881, v894);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %14
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v895 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %14
        ;
        uint64_t v896 = (uint64_t) v70;
        TASSIGN(v895, v896);
        // pto: %194
        ;
        int64_t v897 = v839 < v85 ? v85 : v839;
        // pto: %195
        ;
        const int64_t v898 = 0;
        // pto: %195
        ;
        __gm__ bfloat16_t* v899 = PTOAS__GLOBAL_TENSOR_DATA(v124);
        // pto: %195
        ;
        const int64_t v900 = 1;
        // pto: %195
        ;
        const int64_t v901 = 1;
        // pto: %195
        ;
        const int64_t v902 = 1;
        // pto: %195
        ;
        int64_t v903 = v629 * v76;
        // pto: %195
        ;
        int64_t v904 = v902 * v903;
        // pto: %195
        ;
        pto::Shape<1, 1, 1, -1, 512> v905 = pto::Shape<1, 1, 1, -1, 512>(v900, v901, v902, v629, v88);
        // pto: %195
        ;
        pto::Stride<-1, -1, -1, -1, -1> v906 = pto::Stride<-1, -1, -1, -1, -1>(v901 * v904, v904, v903, v76, v74);
        // pto: %195
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v907 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v899 + (v898 + v659 * v76 + v897 * v74), v905, v906);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        TLOAD(v895, v907);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        // pto: %15
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v908 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %15
        ;
        uint64_t v909 = (uint64_t) v39;
        TASSIGN(v908, v909);
        int64_t v910 = (int64_t) ((uint64_t) v838 + (uint64_t) v36);
        // pto: %198
        ;
        int64_t v911 = v910 < v85 ? v85 : v910;
        // pto: %199
        ;
        const int64_t v912 = 0;
        // pto: %199
        ;
        __gm__ bfloat16_t* v913 = PTOAS__GLOBAL_TENSOR_DATA(v124);
        // pto: %199
        ;
        const int64_t v914 = 1;
        // pto: %199
        ;
        const int64_t v915 = 1;
        // pto: %199
        ;
        const int64_t v916 = 1;
        // pto: %199
        ;
        int64_t v917 = v629 * v76;
        // pto: %199
        ;
        int64_t v918 = v916 * v917;
        // pto: %199
        ;
        pto::Shape<1, 1, 1, -1, 512> v919 = pto::Shape<1, 1, 1, -1, 512>(v914, v915, v916, v629, v88);
        // pto: %199
        ;
        pto::Stride<-1, -1, -1, -1, -1> v920 = pto::Stride<-1, -1, -1, -1, -1>(v915 * v918, v918, v917, v76, v74);
        // pto: %199
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v921 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v913 + (v912 + v659 * v76 + v911 * v74), v919, v920);
        TLOAD(v908, v921);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        // pto: %16
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v922 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %16
        ;
        uint64_t v923 = (uint64_t) v37;
        TASSIGN(v922, v923);
        int64_t v924 = (int64_t) ((uint64_t) v838 + (uint64_t) v35);
        // pto: %202
        ;
        int64_t v925 = v924 < v85 ? v85 : v924;
        // pto: %203
        ;
        const int64_t v926 = 0;
        // pto: %203
        ;
        __gm__ bfloat16_t* v927 = PTOAS__GLOBAL_TENSOR_DATA(v124);
        // pto: %203
        ;
        const int64_t v928 = 1;
        // pto: %203
        ;
        const int64_t v929 = 1;
        // pto: %203
        ;
        const int64_t v930 = 1;
        // pto: %203
        ;
        int64_t v931 = v629 * v76;
        // pto: %203
        ;
        int64_t v932 = v930 * v931;
        // pto: %203
        ;
        pto::Shape<1, 1, 1, -1, 512> v933 = pto::Shape<1, 1, 1, -1, 512>(v928, v929, v930, v629, v88);
        // pto: %203
        ;
        pto::Stride<-1, -1, -1, -1, -1> v934 = pto::Stride<-1, -1, -1, -1, -1>(v929 * v932, v932, v931, v76, v74);
        // pto: %203
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v935 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v927 + (v926 + v659 * v76 + v925 * v74), v933, v934);
        TLOAD(v922, v935);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        // pto: %17
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v936 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %17
        ;
        uint64_t v937 = (uint64_t) v71;
        TASSIGN(v936, v937);
        int64_t v938 = (int64_t) ((uint64_t) v838 + (uint64_t) v34);
        // pto: %206
        ;
        int64_t v939 = v938 < v85 ? v85 : v938;
        // pto: %207
        ;
        const int64_t v940 = 0;
        // pto: %207
        ;
        __gm__ bfloat16_t* v941 = PTOAS__GLOBAL_TENSOR_DATA(v124);
        // pto: %207
        ;
        const int64_t v942 = 1;
        // pto: %207
        ;
        const int64_t v943 = 1;
        // pto: %207
        ;
        const int64_t v944 = 1;
        // pto: %207
        ;
        int64_t v945 = v629 * v76;
        // pto: %207
        ;
        int64_t v946 = v944 * v945;
        // pto: %207
        ;
        pto::Shape<1, 1, 1, -1, 512> v947 = pto::Shape<1, 1, 1, -1, 512>(v942, v943, v944, v629, v88);
        // pto: %207
        ;
        pto::Stride<-1, -1, -1, -1, -1> v948 = pto::Stride<-1, -1, -1, -1, -1>(v943 * v946, v946, v945, v76, v74);
        // pto: %207
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v949 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v941 + (v940 + v659 * v76 + v939 * v74), v947, v948);
        TLOAD(v936, v949);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        // pto: %x0_inline1294__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v950 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %x0_inline1294__ssa_v0
        ;
        uint64_t v951 = (uint64_t) v40;
        TASSIGN(v950, v951);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        RoundMode v952 = RoundMode::CAST_ROUND;
        SaturationMode v953 = SaturationMode::ON;
        TCVT(v950, v840, v952, v953);
        // pto: %x1_inline1293__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v954 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %x1_inline1293__ssa_v0
        ;
        uint64_t v955 = (uint64_t) v67;
        TASSIGN(v954, v955);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        RoundMode v956 = RoundMode::CAST_ROUND;
        SaturationMode v957 = SaturationMode::ON;
        TCVT(v954, v853, v956, v957);
        // pto: %x2_inline1291__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v958 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %x2_inline1291__ssa_v0
        ;
        uint64_t v959 = (uint64_t) v68;
        TASSIGN(v958, v959);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        RoundMode v960 = RoundMode::CAST_ROUND;
        SaturationMode v961 = SaturationMode::ON;
        TCVT(v958, v867, v960, v961);
        // pto: %x3_inline1290__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v962 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %x3_inline1290__ssa_v0
        ;
        uint64_t v963 = (uint64_t) v38;
        TASSIGN(v962, v963);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        RoundMode v964 = RoundMode::CAST_ROUND;
        SaturationMode v965 = SaturationMode::ON;
        TCVT(v962, v881, v964, v965);
        // pto: %y0_inline1469__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v966 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %y0_inline1469__ssa_v0
        ;
        uint64_t v967 = (uint64_t) v40;
        TASSIGN(v966, v967);
        TROWEXPANDMUL(v966, v950, v807);
        // pto: %y1_inline1314__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v968 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %y1_inline1314__ssa_v0
        ;
        uint64_t v969 = (uint64_t) v67;
        TASSIGN(v968, v969);
        TROWEXPANDMUL(v968, v954, v815);
        // pto: %y2_inline1289__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v970 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %y2_inline1289__ssa_v0
        ;
        uint64_t v971 = (uint64_t) v68;
        TASSIGN(v970, v971);
        TROWEXPANDMUL(v970, v958, v823);
        // pto: %y3_inline1520__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v972 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %y3_inline1520__ssa_v0
        ;
        uint64_t v973 = (uint64_t) v38;
        TASSIGN(v972, v973);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v972, v962, v831);
        // pto: %t__tmp_v395
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v974 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %t__tmp_v395
        ;
        uint64_t v975 = (uint64_t) v40;
        TASSIGN(v974, v975);
        TADD(v974, v966, v968);
        // pto: %t__tmp_v396
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v976 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %t__tmp_v396
        ;
        uint64_t v977 = (uint64_t) v67;
        TASSIGN(v976, v977);
        pipe_barrier(PIPE_V);
        TADD(v976, v970, v972);
        // pto: %y_tile_inline1552__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v978 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %y_tile_inline1552__ssa_v0
        ;
        uint64_t v979 = (uint64_t) v40;
        TASSIGN(v978, v979);
        pipe_barrier(PIPE_V);
        TADD(v978, v974, v976);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %y_bf16_inline1590__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v980 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %y_bf16_inline1590__ssa_v0
        ;
        uint64_t v981 = (uint64_t) v40;
        TASSIGN(v980, v981);
        pipe_barrier(PIPE_V);
        RoundMode v982 = RoundMode::CAST_RINT;
        SaturationMode v983 = SaturationMode::ON;
        TCVT(v980, v978, v982, v983);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        v980.SetValidShape(v629, v88);
        // pto: %mixed_view_inline1415__ssa_v0_pview
        ;
        const int64_t v984 = 0;
        // pto: %mixed_view_inline1415__ssa_v0_pview
        ;
        __gm__ bfloat16_t* v985 = PTOAS__GLOBAL_TENSOR_DATA(v185);
        // pto: %mixed_view_inline1415__ssa_v0_pview
        ;
        const int64_t v986 = 1;
        // pto: %mixed_view_inline1415__ssa_v0_pview
        ;
        const int64_t v987 = 1;
        // pto: %mixed_view_inline1415__ssa_v0_pview
        ;
        const int64_t v988 = 1;
        // pto: %mixed_view_inline1415__ssa_v0_pview
        ;
        int64_t v989 = v629 * v80;
        // pto: %mixed_view_inline1415__ssa_v0_pview
        ;
        int64_t v990 = v988 * v989;
        // pto: %mixed_view_inline1415__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, -1, 512> v991 = pto::Shape<1, 1, 1, -1, 512>(v986, v987, v988, v629, v88);
        // pto: %mixed_view_inline1415__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v992 = pto::Stride<-1, -1, -1, -1, -1>(v987 * v990, v990, v989, v80, v74);
        // pto: %mixed_view_inline1415__ssa_v0_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v993 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v985 + (v984 + v659 * v80 + v842 * v74), v991, v992);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v993, v980);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        // pto: %18
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v994 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %18
        ;
        uint64_t v995 = (uint64_t) v72;
        TASSIGN(v994, v995);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
        RoundMode v996 = RoundMode::CAST_ROUND;
        SaturationMode v997 = SaturationMode::ON;
        TCVT(v994, v895, v996, v997);
        // pto: %19
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v998 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %19
        ;
        uint64_t v999 = (uint64_t) v70;
        TASSIGN(v998, v999);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        RoundMode v1000 = RoundMode::CAST_ROUND;
        SaturationMode v1001 = SaturationMode::ON;
        TCVT(v998, v908, v1000, v1001);
        // pto: %20
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1002 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %20
        ;
        uint64_t v1003 = (uint64_t) v39;
        TASSIGN(v1002, v1003);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        RoundMode v1004 = RoundMode::CAST_ROUND;
        SaturationMode v1005 = SaturationMode::ON;
        TCVT(v1002, v922, v1004, v1005);
        // pto: %21
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1006 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %21
        ;
        uint64_t v1007 = (uint64_t) v37;
        TASSIGN(v1006, v1007);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        RoundMode v1008 = RoundMode::CAST_ROUND;
        SaturationMode v1009 = SaturationMode::ON;
        TCVT(v1006, v936, v1008, v1009);
        // pto: %22
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1010 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %22
        ;
        uint64_t v1011 = (uint64_t) v72;
        TASSIGN(v1010, v1011);
        TROWEXPANDMUL(v1010, v994, v807);
        // pto: %23
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1012 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %23
        ;
        uint64_t v1013 = (uint64_t) v70;
        TASSIGN(v1012, v1013);
        TROWEXPANDMUL(v1012, v998, v815);
        // pto: %24
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1014 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %24
        ;
        uint64_t v1015 = (uint64_t) v39;
        TASSIGN(v1014, v1015);
        TROWEXPANDMUL(v1014, v1002, v823);
        // pto: %25
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1016 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %25
        ;
        uint64_t v1017 = (uint64_t) v37;
        TASSIGN(v1016, v1017);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v1016, v1006, v831);
        // pto: %26
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1018 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %26
        ;
        uint64_t v1019 = (uint64_t) v72;
        TASSIGN(v1018, v1019);
        TADD(v1018, v1010, v1012);
        // pto: %27
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1020 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %27
        ;
        uint64_t v1021 = (uint64_t) v70;
        TASSIGN(v1020, v1021);
        pipe_barrier(PIPE_V);
        TADD(v1020, v1014, v1016);
        // pto: %28
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1022 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %28
        ;
        uint64_t v1023 = (uint64_t) v72;
        TASSIGN(v1022, v1023);
        pipe_barrier(PIPE_V);
        TADD(v1022, v1018, v1020);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        // pto: %29
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1024 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %29
        ;
        uint64_t v1025 = (uint64_t) v72;
        TASSIGN(v1024, v1025);
        pipe_barrier(PIPE_V);
        RoundMode v1026 = RoundMode::CAST_RINT;
        SaturationMode v1027 = SaturationMode::ON;
        TCVT(v1024, v1022, v1026, v1027);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        v1024.SetValidShape(v629, v88);
        // pto: %213
        ;
        const int64_t v1028 = 0;
        // pto: %213
        ;
        __gm__ bfloat16_t* v1029 = PTOAS__GLOBAL_TENSOR_DATA(v185);
        // pto: %213
        ;
        const int64_t v1030 = 1;
        // pto: %213
        ;
        const int64_t v1031 = 1;
        // pto: %213
        ;
        const int64_t v1032 = 1;
        // pto: %213
        ;
        int64_t v1033 = v629 * v80;
        // pto: %213
        ;
        int64_t v1034 = v1032 * v1033;
        // pto: %213
        ;
        pto::Shape<1, 1, 1, -1, 512> v1035 = pto::Shape<1, 1, 1, -1, 512>(v1030, v1031, v1032, v629, v88);
        // pto: %213
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1036 = pto::Stride<-1, -1, -1, -1, -1>(v1031 * v1034, v1034, v1033, v80, v74);
        // pto: %213
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1037 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1029 + (v1028 + v659 * v80 + v897 * v74), v1035, v1036);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v1037, v1024);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
      };
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      // pto: %214
      ;
      uint64_t v1038 = (uint64_t) v602;
      // pto: %214, %215
      ;
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      if ((int64_t) (v1038 - (uint64_t) v836) == v74) {
        // pto: %216
        ;
        int64_t v1039 = (int64_t) ((uint64_t) v836 * (uint64_t) v88);
        // pto: %31
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1040 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %31
        ;
        uint64_t v1041 = (uint64_t) v67;
        TASSIGN(v1040, v1041);
        // pto: %219
        ;
        int64_t v1042 = v1039 < v85 ? v85 : v1039;
        // pto: %220
        ;
        const int64_t v1043 = 0;
        // pto: %220
        ;
        __gm__ bfloat16_t* v1044 = PTOAS__GLOBAL_TENSOR_DATA(v124);
        // pto: %220
        ;
        const int64_t v1045 = 1;
        // pto: %220
        ;
        const int64_t v1046 = 1;
        // pto: %220
        ;
        const int64_t v1047 = 1;
        // pto: %220
        ;
        int64_t v1048 = v629 * v76;
        // pto: %220
        ;
        int64_t v1049 = v1047 * v1048;
        // pto: %220
        ;
        pto::Shape<1, 1, 1, -1, 512> v1050 = pto::Shape<1, 1, 1, -1, 512>(v1045, v1046, v1047, v629, v88);
        // pto: %220
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1051 = pto::Stride<-1, -1, -1, -1, -1>(v1046 * v1049, v1049, v1048, v76, v74);
        // pto: %220
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1052 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1044 + (v1043 + v659 * v76 + v1042 * v74), v1050, v1051);
        TLOAD(v1040, v1052);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %32
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1053 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %32
        ;
        uint64_t v1054 = (uint64_t) v40;
        TASSIGN(v1053, v1054);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v1055 = RoundMode::CAST_ROUND;
        SaturationMode v1056 = SaturationMode::ON;
        TCVT(v1053, v1040, v1055, v1056);
        // pto: %33
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1057 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %33
        ;
        uint64_t v1058 = (uint64_t) v68;
        TASSIGN(v1057, v1058);
        // pto: %222
        ;
        int64_t v1059 = (int64_t) ((uint64_t) v1039 + (uint64_t) v80);
        // pto: %223
        ;
        int64_t v1060 = v1059 < v85 ? v85 : v1059;
        // pto: %224
        ;
        const int64_t v1061 = 0;
        // pto: %224
        ;
        __gm__ bfloat16_t* v1062 = PTOAS__GLOBAL_TENSOR_DATA(v124);
        // pto: %224
        ;
        const int64_t v1063 = 1;
        // pto: %224
        ;
        const int64_t v1064 = 1;
        // pto: %224
        ;
        const int64_t v1065 = 1;
        // pto: %224
        ;
        int64_t v1066 = v629 * v76;
        // pto: %224
        ;
        int64_t v1067 = v1065 * v1066;
        // pto: %224
        ;
        pto::Shape<1, 1, 1, -1, 512> v1068 = pto::Shape<1, 1, 1, -1, 512>(v1063, v1064, v1065, v629, v88);
        // pto: %224
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1069 = pto::Stride<-1, -1, -1, -1, -1>(v1064 * v1067, v1067, v1066, v76, v74);
        // pto: %224
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1070 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1062 + (v1061 + v659 * v76 + v1060 * v74), v1068, v1069);
        TLOAD(v1057, v1070);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %34
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1071 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %34
        ;
        uint64_t v1072 = (uint64_t) v67;
        TASSIGN(v1071, v1072);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v1073 = RoundMode::CAST_ROUND;
        SaturationMode v1074 = SaturationMode::ON;
        TCVT(v1071, v1057, v1073, v1074);
        // pto: %35
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1075 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %35
        ;
        uint64_t v1076 = (uint64_t) v38;
        TASSIGN(v1075, v1076);
        // pto: %226
        ;
        int64_t v1077 = (int64_t) ((uint64_t) v1039 + (uint64_t) v105);
        // pto: %227
        ;
        int64_t v1078 = v1077 < v85 ? v85 : v1077;
        // pto: %228
        ;
        const int64_t v1079 = 0;
        // pto: %228
        ;
        __gm__ bfloat16_t* v1080 = PTOAS__GLOBAL_TENSOR_DATA(v124);
        // pto: %228
        ;
        const int64_t v1081 = 1;
        // pto: %228
        ;
        const int64_t v1082 = 1;
        // pto: %228
        ;
        const int64_t v1083 = 1;
        // pto: %228
        ;
        int64_t v1084 = v629 * v76;
        // pto: %228
        ;
        int64_t v1085 = v1083 * v1084;
        // pto: %228
        ;
        pto::Shape<1, 1, 1, -1, 512> v1086 = pto::Shape<1, 1, 1, -1, 512>(v1081, v1082, v1083, v629, v88);
        // pto: %228
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1087 = pto::Stride<-1, -1, -1, -1, -1>(v1082 * v1085, v1085, v1084, v76, v74);
        // pto: %228
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1088 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1080 + (v1079 + v659 * v76 + v1078 * v74), v1086, v1087);
        TLOAD(v1075, v1088);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %36
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1089 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %36
        ;
        uint64_t v1090 = (uint64_t) v68;
        TASSIGN(v1089, v1090);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v1091 = RoundMode::CAST_ROUND;
        SaturationMode v1092 = SaturationMode::ON;
        TCVT(v1089, v1075, v1091, v1092);
        // pto: %37
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1093 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %37
        ;
        uint64_t v1094 = (uint64_t) v72;
        TASSIGN(v1093, v1094);
        // pto: %230
        ;
        int64_t v1095 = (int64_t) ((uint64_t) v1039 + (uint64_t) v106);
        // pto: %231
        ;
        int64_t v1096 = v1095 < v85 ? v85 : v1095;
        // pto: %232
        ;
        const int64_t v1097 = 0;
        // pto: %232
        ;
        __gm__ bfloat16_t* v1098 = PTOAS__GLOBAL_TENSOR_DATA(v124);
        // pto: %232
        ;
        const int64_t v1099 = 1;
        // pto: %232
        ;
        const int64_t v1100 = 1;
        // pto: %232
        ;
        const int64_t v1101 = 1;
        // pto: %232
        ;
        int64_t v1102 = v629 * v76;
        // pto: %232
        ;
        int64_t v1103 = v1101 * v1102;
        // pto: %232
        ;
        pto::Shape<1, 1, 1, -1, 512> v1104 = pto::Shape<1, 1, 1, -1, 512>(v1099, v1100, v1101, v629, v88);
        // pto: %232
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1105 = pto::Stride<-1, -1, -1, -1, -1>(v1100 * v1103, v1103, v1102, v76, v74);
        // pto: %232
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1106 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1098 + (v1097 + v659 * v76 + v1096 * v74), v1104, v1105);
        TLOAD(v1093, v1106);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %38
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1107 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %38
        ;
        uint64_t v1108 = (uint64_t) v38;
        TASSIGN(v1107, v1108);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v1109 = RoundMode::CAST_ROUND;
        SaturationMode v1110 = SaturationMode::ON;
        TCVT(v1107, v1093, v1109, v1110);
        // pto: %39
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1111 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %39
        ;
        uint64_t v1112 = (uint64_t) v40;
        TASSIGN(v1111, v1112);
        TROWEXPANDMUL(v1111, v1053, v807);
        // pto: %40
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1113 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %40
        ;
        uint64_t v1114 = (uint64_t) v67;
        TASSIGN(v1113, v1114);
        TROWEXPANDMUL(v1113, v1071, v815);
        // pto: %41
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1115 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %41
        ;
        uint64_t v1116 = (uint64_t) v68;
        TASSIGN(v1115, v1116);
        TROWEXPANDMUL(v1115, v1089, v823);
        // pto: %42
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1117 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %42
        ;
        uint64_t v1118 = (uint64_t) v38;
        TASSIGN(v1117, v1118);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v1117, v1107, v831);
        // pto: %43
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1119 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %43
        ;
        uint64_t v1120 = (uint64_t) v40;
        TASSIGN(v1119, v1120);
        TADD(v1119, v1111, v1113);
        // pto: %44
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1121 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %44
        ;
        uint64_t v1122 = (uint64_t) v67;
        TASSIGN(v1121, v1122);
        pipe_barrier(PIPE_V);
        TADD(v1121, v1115, v1117);
        // pto: %45
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1123 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %45
        ;
        uint64_t v1124 = (uint64_t) v40;
        TASSIGN(v1123, v1124);
        pipe_barrier(PIPE_V);
        TADD(v1123, v1119, v1121);
        // pto: %46
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1125 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v629, v88);
        // pto: %46
        ;
        uint64_t v1126 = (uint64_t) v40;
        TASSIGN(v1125, v1126);
        pipe_barrier(PIPE_V);
        RoundMode v1127 = RoundMode::CAST_RINT;
        SaturationMode v1128 = SaturationMode::ON;
        TCVT(v1125, v1123, v1127, v1128);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        v1125.SetValidShape(v629, v88);
        // pto: %236
        ;
        const int64_t v1129 = 0;
        // pto: %236
        ;
        __gm__ bfloat16_t* v1130 = PTOAS__GLOBAL_TENSOR_DATA(v185);
        // pto: %236
        ;
        const int64_t v1131 = 1;
        // pto: %236
        ;
        const int64_t v1132 = 1;
        // pto: %236
        ;
        const int64_t v1133 = 1;
        // pto: %236
        ;
        int64_t v1134 = v629 * v80;
        // pto: %236
        ;
        int64_t v1135 = v1133 * v1134;
        // pto: %236
        ;
        pto::Shape<1, 1, 1, -1, 512> v1136 = pto::Shape<1, 1, 1, -1, 512>(v1131, v1132, v1133, v629, v88);
        // pto: %236
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1137 = pto::Stride<-1, -1, -1, -1, -1>(v1132 * v1135, v1135, v1134, v80, v74);
        // pto: %236
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1138 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1130 + (v1129 + v659 * v80 + v1042 * v74), v1136, v1137);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v1138, v1125);
      };
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    };
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  }
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}