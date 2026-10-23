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

AICORE void hc_large_aic(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ bfloat16_t* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, __gm__ float* v10, __gm__ float* v11, __gm__ bfloat16_t* v12, __gm__ float* v13, int64_t v14, int32_t v15, int64_t v16, int64_t v17, int64_t v18, int64_t v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int64_t v28, int64_t v29, int64_t v30, int32_t v31, int32_t v32) {
  using T = float;

  #if defined(__DAV_CUBE__)
  const int64_t v33 = 384;
  const int32_t v34 = 0;
  // pto: %c0_i64
  const int64_t v35 = 0;
  // pto: %c262144_i64
  const int64_t v36 = 262144;
  // pto: %c327680_i64
  const int64_t v37 = 327680;
  // pto: %c32768_i64
  const int64_t v38 = 32768;
  // pto: %c16384_i64
  const int64_t v39 = 16384;
  // pto: %c49152_i64
  const int64_t v40 = 49152;
  // pto: %c32_index
  const int64_t v41 = 32;
  // pto: %c1_index
  const int64_t v42 = 1;
  // pto: %c24_index
  const int64_t v43 = 24;
  // pto: %c16384_index
  const int64_t v44 = 16384;
  // pto: %c8_index
  const int64_t v45 = 8;
  // pto: %c64_index
  const int64_t v46 = 64;
  // pto: %c2048_index
  const int64_t v47 = 2048;
  // pto: %c0_index
  const int64_t v48 = 0;
  // pto: %c4_index
  const int64_t v49 = 4;
  // pto: %c2_index
  const int64_t v50 = 2;
  // pto: %c512_index
  const int64_t v51 = 512;
  // pto: %c256_index
  const int64_t v52 = 256;
  // pto: %c128_index
  const int64_t v53 = 128;
  // pto: %cn128_index
  const int64_t v54 = -128;
  // pto: %63
  int64_t v55 = (int64_t) ((uint64_t) v29 * (uint64_t) v45);
  // pto: %mixes_partials_inline1724__ssa_v0_view
  const int64_t v56 = 1;
  // pto: %mixes_partials_inline1724__ssa_v0_view
  const int64_t v57 = 1;
  // pto: %mixes_partials_inline1724__ssa_v0_view
  const int64_t v58 = 1;
  // pto: %mixes_partials_inline1724__ssa_v0_view
  int64_t v59 = v55 * v41;
  // pto: %mixes_partials_inline1724__ssa_v0_view
  int64_t v60 = v58 * v59;
  // pto: %mixes_partials_inline1724__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v61 = pto::Shape<1, 1, 1, -1, -1>(v56, v57, v58, v55, v41);
  // pto: %mixes_partials_inline1724__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v62 = pto::Stride<-1, -1, -1, -1, -1>(v57 * v60, v60, v59, v41, v42);
  // pto: %mixes_partials_inline1724__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v63 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v61, v62);
  // pto: %hc_fn__ssa_v0_view
  const int64_t v64 = 1;
  // pto: %hc_fn__ssa_v0_view
  const int64_t v65 = 1;
  // pto: %hc_fn__ssa_v0_view
  const int64_t v66 = 1;
  // pto: %hc_fn__ssa_v0_view
  int64_t v67 = v43 * v44;
  // pto: %hc_fn__ssa_v0_view
  int64_t v68 = v66 * v67;
  // pto: %hc_fn__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v69 = pto::Shape<1, 1, 1, -1, -1>(v64, v65, v66, v43, v44);
  // pto: %hc_fn__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v70 = pto::Stride<-1, -1, -1, -1, -1>(v65 * v68, v68, v67, v44, v42);
  // pto: %hc_fn__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v71 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v69, v70);
  // pto: %mid_stream_product_inline1686__phi_v5
  Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v72 = Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v46, v43);
  // pto: %mid_stream_product_inline1686__phi_v5
  uint64_t v73 = (uint64_t) v35;
  TASSIGN(v72, v73);
  const int32_t v74 = 0;
  auto v75 = TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>(v13, v74, v34);
  // pto: %worker_inline1650__ssa_v0
  // pto: %linear_workers_inline1748__ssa_v0_idx
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
  for (int64_t v76 = (int64_t) v31; v76 < v14; v76 += (int64_t) v15) {
    // pto: %20
    ;
    int64_t v77 = v76 % v45;
    // pto: %21
    ;
    int64_t v78 = (int64_t) ((uint64_t) v77 * (uint64_t) v47);
    // pto: %22, %23, %18, %19
    ;
    int64_t v79 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v77 * (uint64_t) v16) + (uint64_t) ((int64_t) (uint64_t) (v76 / v45) * (uint64_t) v46));
    // pto: %mid_stream_product_inline1686__tile
    ;
    Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v80 = Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v46, v41);
    // pto: %mid_stream_product_inline1686__tile
    ;
    uint64_t v81 = (uint64_t) v35;
    TASSIGN(v80, v81);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int64_t v82 = v48; v82 < v49; v82 += v50) {
      // pto: %24
      ;
      int64_t v83 = (int64_t) ((uint64_t) v82 * (uint64_t) v51);
      // pto: %25
      ;
      int64_t v84 = (int64_t) ((uint64_t) v78 + (uint64_t) v83);
      // pto: %28, %27
      ;
      int64_t v85 = (int64_t) ((uint64_t) v78 + (uint64_t) ((int64_t) (uint64_t) v83 + (uint64_t) v51));
      // pto: %mid_weight_chunk_inline1677__tile
      ;
      Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v86 = Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v43, v51);
      // pto: %mid_weight_chunk_inline1677__tile
      ;
      uint64_t v87 = (uint64_t) v36;
      TASSIGN(v86, v87);
      // pto: %29
      ;
      int64_t v88 = v84 < v48 ? v48 : v84;
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v89 = 0;
      // pto: %hc_fn__ssa_v0_pview
      ;
      __gm__ float* v90 = PTOAS__GLOBAL_TENSOR_DATA(v71);
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v91 = 1;
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v92 = 1;
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v93 = 1;
      // pto: %hc_fn__ssa_v0_pview
      ;
      int64_t v94 = v43 * v44;
      // pto: %hc_fn__ssa_v0_pview
      ;
      int64_t v95 = v93 * v94;
      // pto: %hc_fn__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 24, -1> v96 = pto::Shape<1, 1, 1, 24, -1>(v91, v92, v93, v43, v51);
      // pto: %hc_fn__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v97 = pto::Stride<-1, -1, -1, -1, -1>(v92 * v95, v95, v94, v44, v42);
      // pto: %hc_fn__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v98 = GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v90 + (v89 + v48 * v44 + v88 * v42), v96, v97);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v86, v98);
      // pto: %0
      ;
      Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v99 = Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v43, v51);
      // pto: %0
      ;
      uint64_t v100 = (uint64_t) v37;
      TASSIGN(v99, v100);
      // pto: %30
      ;
      int64_t v101 = v85 < v48 ? v48 : v85;
      // pto: %31
      ;
      const int64_t v102 = 0;
      // pto: %31
      ;
      __gm__ float* v103 = PTOAS__GLOBAL_TENSOR_DATA(v71);
      // pto: %31
      ;
      const int64_t v104 = 1;
      // pto: %31
      ;
      const int64_t v105 = 1;
      // pto: %31
      ;
      const int64_t v106 = 1;
      // pto: %31
      ;
      int64_t v107 = v43 * v44;
      // pto: %31
      ;
      int64_t v108 = v106 * v107;
      // pto: %31
      ;
      pto::Shape<1, 1, 1, 24, -1> v109 = pto::Shape<1, 1, 1, 24, -1>(v104, v105, v106, v43, v51);
      // pto: %31
      ;
      pto::Stride<-1, -1, -1, -1, -1> v110 = pto::Stride<-1, -1, -1, -1, -1>(v105 * v108, v108, v107, v44, v42);
      // pto: %31
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v111 = GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v103 + (v102 + v48 * v44 + v101 * v42), v109, v110);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      TLOAD(v99, v111);
      // pto: %mid_gathered_chunk_inline1671__tile
      ;
      const int32_t v112 = 64;
      // pto: %mid_gathered_chunk_inline1671__tile
      ;
      const int32_t v113 = 512;
      // pto: %mid_gathered_chunk_inline1671__tile
      ;
      Tile<TileType::Mat, float, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v114 = Tile<TileType::Mat, float, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v112, v113);
      pipe_barrier(PIPE_MTE2);
      TPOP<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, Tile<TileType::Mat, float, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v75, v114);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %32
      ;
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      if (v82 == v48) {
        // pto: %mid_weight_chunk_inline1677__tile_t
        ;
        Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v115 = Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v51, v43);
        // pto: %mid_weight_chunk_inline1677__tile_t
        ;
        uint64_t v116 = (uint64_t) v36;
        TASSIGN(v115, v116);
        // pto: %mid_stream_product_inline1686__tile_l0_init_storage
        ;
        Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v117 = Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v46, v41);
        // pto: %mid_stream_product_inline1686__tile_l0_init_storage
        ;
        uint64_t v118 = (uint64_t) v35;
        TASSIGN(v117, v118);
        v117.SetValidShape(v46, v43);
        for (int64_t v119 = v48; v119 < v51; v119 += v52) {
          // pto: %mid_stream_product_inline1686__tile_l0_a
          ;
          Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v120 = Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v46, v53);
          // pto: %mid_stream_product_inline1686__tile_l0_a
          ;
          uint64_t v121 = (uint64_t) v35;
          TASSIGN(v120, v121);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          pipe_barrier(PIPE_MTE1);
          uint16_t v122 = static_cast<uint16_t>(v48);
          uint16_t v123 = static_cast<uint16_t>(v119);
          TEXTRACT(v120, v114, v122, v123);
          // pto: %33
          ;
          int64_t v124 = (int64_t) ((uint64_t) v51 - (uint64_t) v119);
          // pto: %34
          ;
          int64_t v125 = v124 < v53 ? v124 : v53;
          // pto: %mid_stream_product_inline1686__tile_l0_b
          ;
          Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v126 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v125, v43);
          // pto: %mid_stream_product_inline1686__tile_l0_b
          ;
          uint64_t v127 = (uint64_t) v35;
          TASSIGN(v126, v127);
          uint16_t v128 = static_cast<uint16_t>(v119);
          uint16_t v129 = static_cast<uint16_t>(v48);
          TEXTRACT(v126, v115, v128, v129);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          // pto: %1
          ;
          Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v130 = Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v46, v53);
          // pto: %1
          ;
          uint64_t v131 = (uint64_t) v38;
          TASSIGN(v130, v131);
          // pto: %35
          ;
          int64_t v132 = (int64_t) ((uint64_t) v119 + (uint64_t) v53);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
          uint16_t v133 = static_cast<uint16_t>(v48);
          uint16_t v134 = static_cast<uint16_t>(v132);
          TEXTRACT(v130, v114, v133, v134);
          int64_t v135 = (int64_t) ((uint64_t) v33 - (uint64_t) v119);
          // pto: %38
          ;
          int64_t v136 = v135 < v48 ? v48 : v135;
          // pto: %39
          ;
          int64_t v137 = v136 < v53 ? v136 : v53;
          // pto: %2
          ;
          Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v138 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v137, v43);
          // pto: %2
          ;
          uint64_t v139 = (uint64_t) v39;
          TASSIGN(v138, v139);
          uint16_t v140 = static_cast<uint16_t>(v132);
          uint16_t v141 = static_cast<uint16_t>(v48);
          TEXTRACT(v138, v115, v140, v141);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          // pto: %41
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          if (v119 == v48) {
            pipe_barrier(PIPE_M);
            TMATMUL(v117, v120, v126);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v117, v117, v120, v126);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          // pto: %42
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          if (v119 == v54) {
            pipe_barrier(PIPE_M);
            TMATMUL(v117, v130, v138);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v117, v117, v130, v138);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        };
      } else {
        // pto: %4
        ;
        Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v142 = Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v51, v43);
        // pto: %4
        ;
        uint64_t v143 = (uint64_t) v36;
        TASSIGN(v142, v143);
        for (int64_t v144 = v48; v144 < v51; v144 += v52) {
          // pto: %5
          ;
          Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v145 = Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v46, v53);
          // pto: %5
          ;
          uint64_t v146 = (uint64_t) v35;
          TASSIGN(v145, v146);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
          uint16_t v147 = static_cast<uint16_t>(v48);
          uint16_t v148 = static_cast<uint16_t>(v144);
          TEXTRACT(v145, v114, v147, v148);
          // pto: %44
          ;
          int64_t v149 = (int64_t) ((uint64_t) v51 - (uint64_t) v144);
          // pto: %45
          ;
          int64_t v150 = v149 < v53 ? v149 : v53;
          // pto: %6
          ;
          Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v151 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v150, v43);
          // pto: %6
          ;
          uint64_t v152 = (uint64_t) v35;
          TASSIGN(v151, v152);
          uint16_t v153 = static_cast<uint16_t>(v144);
          uint16_t v154 = static_cast<uint16_t>(v48);
          TEXTRACT(v151, v142, v153, v154);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
          // pto: %7
          ;
          Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v155 = Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v46, v53);
          // pto: %7
          ;
          uint64_t v156 = (uint64_t) v38;
          TASSIGN(v155, v156);
          // pto: %46
          ;
          int64_t v157 = (int64_t) ((uint64_t) v144 + (uint64_t) v53);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
          uint16_t v158 = static_cast<uint16_t>(v48);
          uint16_t v159 = static_cast<uint16_t>(v157);
          TEXTRACT(v155, v114, v158, v159);
          int64_t v160 = (int64_t) ((uint64_t) v33 - (uint64_t) v144);
          // pto: %49
          ;
          int64_t v161 = v160 < v48 ? v48 : v160;
          // pto: %50
          ;
          int64_t v162 = v161 < v53 ? v161 : v53;
          // pto: %8
          ;
          Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v163 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v162, v43);
          // pto: %8
          ;
          uint64_t v164 = (uint64_t) v39;
          TASSIGN(v163, v164);
          uint16_t v165 = static_cast<uint16_t>(v157);
          uint16_t v166 = static_cast<uint16_t>(v48);
          TEXTRACT(v163, v142, v165, v166);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
          pipe_barrier(PIPE_M);
          TMATMUL_ACC(v80, v80, v145, v151);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
          pipe_barrier(PIPE_M);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
          TMATMUL_ACC(v80, v80, v155, v163);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        };
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TFREE<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, TileSplitAxis::TILE_UP_DOWN>(v75);
      // pto: %52
      ;
      const int32_t v167 = 64;
      // pto: %52
      ;
      const int32_t v168 = 512;
      // pto: %52
      ;
      Tile<TileType::Mat, float, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v169 = Tile<TileType::Mat, float, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v167, v168);
      TPOP<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, Tile<TileType::Mat, float, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v75, v169);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      // pto: %11
      ;
      Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v170 = Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v51, v43);
      // pto: %11
      ;
      uint64_t v171 = (uint64_t) v37;
      TASSIGN(v170, v171);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
      for (int64_t v172 = v48; v172 < v51; v172 += v52) {
        // pto: %12
        ;
        Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v173 = Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v46, v53);
        // pto: %12
        ;
        uint64_t v174 = (uint64_t) v35;
        TASSIGN(v173, v174);
        pipe_barrier(PIPE_MTE1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
        uint16_t v175 = static_cast<uint16_t>(v48);
        uint16_t v176 = static_cast<uint16_t>(v172);
        TEXTRACT(v173, v169, v175, v176);
        // pto: %54
        ;
        int64_t v177 = (int64_t) ((uint64_t) v51 - (uint64_t) v172);
        // pto: %55
        ;
        int64_t v178 = v177 < v53 ? v177 : v53;
        // pto: %13
        ;
        Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v179 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v178, v43);
        // pto: %13
        ;
        uint64_t v180 = (uint64_t) v38;
        TASSIGN(v179, v180);
        uint16_t v181 = static_cast<uint16_t>(v172);
        uint16_t v182 = static_cast<uint16_t>(v48);
        TEXTRACT(v179, v170, v181, v182);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
        // pto: %14
        ;
        Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v183 = Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v46, v53);
        // pto: %14
        ;
        uint64_t v184 = (uint64_t) v38;
        TASSIGN(v183, v184);
        // pto: %56
        ;
        int64_t v185 = (int64_t) ((uint64_t) v172 + (uint64_t) v53);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
        uint16_t v186 = static_cast<uint16_t>(v48);
        uint16_t v187 = static_cast<uint16_t>(v185);
        TEXTRACT(v183, v169, v186, v187);
        int64_t v188 = (int64_t) ((uint64_t) v33 - (uint64_t) v172);
        // pto: %59
        ;
        int64_t v189 = v188 < v48 ? v48 : v188;
        // pto: %60
        ;
        int64_t v190 = v189 < v53 ? v189 : v53;
        // pto: %15
        ;
        Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v191 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v190, v43);
        // pto: %15
        ;
        uint64_t v192 = (uint64_t) v40;
        TASSIGN(v191, v192);
        uint16_t v193 = static_cast<uint16_t>(v185);
        uint16_t v194 = static_cast<uint16_t>(v48);
        TEXTRACT(v191, v170, v193, v194);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v72, v72, v173, v179);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
        TMATMUL_ACC(v72, v72, v183, v191);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      TFREE<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, TileSplitAxis::TILE_UP_DOWN>(v75);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    // pto: %62
    ;
    int64_t v195 = v79 < v48 ? v48 : v79;
    // pto: %mixes_partials_inline1724__iter_v1_pview
    ;
    __gm__ float* v196 = PTOAS__GLOBAL_TENSOR_DATA(v63);
    // pto: %mixes_partials_inline1724__iter_v1_pview
    ;
    const int64_t v197 = 0;
    // pto: %mixes_partials_inline1724__iter_v1_pview
    ;
    const int64_t v198 = 32;
    // pto: %mixes_partials_inline1724__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 64, 32> v199 = pto::Shape<1, 1, 1, 64, 32>();
    // pto: %mixes_partials_inline1724__iter_v1_pview
    ;
    pto::Stride<2048, 2048, 2048, 32, 1> v200 = pto::Stride<2048, 2048, 2048, 32, 1>();
    // pto: %mixes_partials_inline1724__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 64, 32>, pto::Stride<2048, 2048, 2048, 32, 1>, pto::Layout::ND> v201 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 32>, pto::Stride<2048, 2048, 2048, 32, 1>, pto::Layout::ND>(v196 + (v197 + v195 * v198), v199, v200);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v201, v80);
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

AICORE void hc_large_aiv(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ bfloat16_t* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, __gm__ float* v10, __gm__ float* v11, __gm__ bfloat16_t* v12, __gm__ float* v13, int64_t v14, int32_t v15, int64_t v16, int64_t v17, int64_t v18, int64_t v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int64_t v28, int64_t v29, int64_t v30, int32_t v31, int32_t v32, int32_t v33) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v34 = 0;
  // pto: %c98432_i64
  const int64_t v35 = 98432;
  // pto: %c128_i64
  const int64_t v36 = 128;
  // pto: %c32896_i64
  const int64_t v37 = 32896;
  // pto: %c163968_i64
  const int64_t v38 = 163968;
  // pto: %c41088_i64
  const int64_t v39 = 41088;
  // pto: %c33920_i64
  const int64_t v40 = 33920;
  // pto: %c34176_i64
  const int64_t v41 = 34176;
  // pto: %c34432_i64
  const int64_t v42 = 34432;
  // pto: %c33952_i64
  const int64_t v43 = 33952;
  const int64_t v44 = 32928;
  // pto: %c33408_i64
  const int64_t v45 = 33408;
  // pto: %c33664_i64
  const int64_t v46 = 33664;
  // pto: %c34688_i64
  const int64_t v47 = 34688;
  // pto: %c35200_i64
  const int64_t v48 = 35200;
  // pto: %c35712_i64
  const int64_t v49 = 35712;
  // pto: %c36224_i64
  const int64_t v50 = 36224;
  // pto: %c36736_i64
  const int64_t v51 = 36736;
  // pto: %c37248_i64
  const int64_t v52 = 37248;
  // pto: %c34944_i64
  const int64_t v53 = 34944;
  // pto: %c34720_i64
  const int64_t v54 = 34720;
  // pto: %c34752_i64
  const int64_t v55 = 34752;
  // pto: %c34784_i64
  const int64_t v56 = 34784;
  // pto: %c33984_i64
  const int64_t v57 = 33984;
  // pto: %c33152_i64
  const int64_t v58 = 33152;
  // pto: %c147584_i64
  const int64_t v59 = 147584;
  // pto: %c33024_i64
  const int64_t v60 = 33024;
  // pto: %c49280_i64
  const int64_t v61 = 49280;
  // pto: %c65664_i64
  const int64_t v62 = 65664;
  // pto: %c73856_i64
  const int64_t v63 = 73856;
  // pto: %c131200_i64
  const int64_t v64 = 131200;
  // pto: %c114816_i64
  const int64_t v65 = 114816;
  // pto: %c123008_i64
  const int64_t v66 = 123008;
  // pto: %c139392_i64
  const int64_t v67 = 139392;
  // pto: %c82048_i64
  const int64_t v68 = 82048;
  // pto: %c32_index
  const int64_t v69 = 32;
  // pto: %c1_index
  const int64_t v70 = 1;
  // pto: %c24_index
  const int64_t v71 = 24;
  // pto: %c16384_index
  const int64_t v72 = 16384;
  // pto: %c3_index
  const int64_t v73 = 3;
  // pto: %c8_index
  const int64_t v74 = 8;
  const int64_t v75 = 16;
  // pto: %c4096_index
  const int64_t v76 = 4096;
  // pto: %c0_i32
  const int32_t v77 = 0;
  // pto: %c64_index
  const int64_t v78 = 64;
  // pto: %c2048_index
  const int64_t v79 = 2048;
  // pto: %cst_46
  const float v80 = 0.0f;
  // pto: %c0_index
  const int64_t v81 = 0;
  // pto: %c4_index
  const int64_t v82 = 4;
  // pto: %c2_index
  const int64_t v83 = 2;
  // pto: %c512_index
  const int64_t v84 = 512;
  // pto: %c256_index
  const int64_t v85 = 256;
  // pto: %cst_52
  const float v86 = 6.10351563E-5f;
  // pto: %cst_53
  const float v87 = 9.99999997E-7f;
  // pto: %c128_index
  const int64_t v88 = 128;
  // pto: %c192_index
  const int64_t v89 = 192;
  // pto: %c1_i32
  const int32_t v90 = 1;
  // pto: %c2_i32
  const int32_t v91 = 2;
  // pto: %c4_i32
  const int32_t v92 = 4;
  // pto: %c8_i32
  const int32_t v93 = 8;
  // pto: %c19_index
  const int64_t v94 = 19;
  // pto: %cst_61
  const float v95 = 0.25f;
  // pto: %cst_62
  const float v96 = 1.0f;
  // pto: %cst_63
  const float v97 = 4.0f;
  // pto: %cst_64
  const float v98 = 2.0f;
  // pto: %243
  int64_t v99 = (int64_t) ((uint64_t) v29 * (uint64_t) v74);
  // pto: %rms_partials_inline1725__ssa_v0_view
  const int64_t v100 = 1;
  // pto: %rms_partials_inline1725__ssa_v0_view
  const int64_t v101 = 1;
  // pto: %rms_partials_inline1725__ssa_v0_view
  const int64_t v102 = 1;
  // pto: %rms_partials_inline1725__ssa_v0_view
  int64_t v103 = v99 * v70;
  // pto: %rms_partials_inline1725__ssa_v0_view
  int64_t v104 = v102 * v103;
  // pto: %rms_partials_inline1725__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v105 = pto::Shape<1, 1, 1, -1, -1>(v100, v101, v102, v99, v70);
  // pto: %rms_partials_inline1725__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v106 = pto::Stride<-1, -1, -1, -1, -1>(v101 * v104, v104, v103, v70, v99);
  // pto: %rms_partials_inline1725__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v107 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v105, v106);
  // pto: %x_flat_inline1654__ssa_v0_view
  const int64_t v108 = 1;
  // pto: %x_flat_inline1654__ssa_v0_view
  const int64_t v109 = 1;
  // pto: %x_flat_inline1654__ssa_v0_view
  const int64_t v110 = 1;
  // pto: %x_flat_inline1654__ssa_v0_view
  int64_t v111 = (int64_t) v30;
  // pto: %x_flat_inline1654__ssa_v0_view
  int64_t v112 = v111 * v72;
  // pto: %x_flat_inline1654__ssa_v0_view
  int64_t v113 = v110 * v112;
  // pto: %x_flat_inline1654__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v114 = pto::Shape<1, 1, 1, -1, -1>(v108, v109, v110, v111, v72);
  // pto: %x_flat_inline1654__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v115 = pto::Stride<-1, -1, -1, -1, -1>(v109 * v113, v113, v112, v72, v70);
  // pto: %x_flat_inline1654__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v116 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v114, v115);
  // pto: %244
  int64_t v117 = (int64_t) ((uint64_t) v29 * (uint64_t) v69);
  // pto: %parts_2d_inline1739__ssa_v0_view
  const int64_t v118 = 1;
  // pto: %parts_2d_inline1739__ssa_v0_view
  const int64_t v119 = 1;
  // pto: %parts_2d_inline1739__ssa_v0_view
  const int64_t v120 = 1;
  // pto: %parts_2d_inline1739__ssa_v0_view
  int64_t v121 = v74 * v117;
  // pto: %parts_2d_inline1739__ssa_v0_view
  int64_t v122 = v120 * v121;
  // pto: %parts_2d_inline1739__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v123 = pto::Shape<1, 1, 1, -1, -1>(v118, v119, v120, v74, v117);
  // pto: %parts_2d_inline1739__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v124 = pto::Stride<-1, -1, -1, -1, -1>(v119 * v122, v122, v121, v117, v70);
  // pto: %parts_2d_inline1739__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v125 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v123, v124);
  // pto: %rms_2d_inline1751__ssa_v0_view
  const int64_t v126 = 1;
  // pto: %rms_2d_inline1751__ssa_v0_view
  const int64_t v127 = 1;
  // pto: %rms_2d_inline1751__ssa_v0_view
  const int64_t v128 = 1;
  // pto: %rms_2d_inline1751__ssa_v0_view
  int64_t v129 = (int64_t) v29;
  // pto: %rms_2d_inline1751__ssa_v0_view
  int64_t v130 = v74 * v129;
  // pto: %rms_2d_inline1751__ssa_v0_view
  int64_t v131 = v128 * v130;
  // pto: %rms_2d_inline1751__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v132 = pto::Shape<1, 1, 1, -1, -1>(v126, v127, v128, v74, (int64_t) v29);
  // pto: %rms_2d_inline1751__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v133 = pto::Stride<-1, -1, -1, -1, -1>(v127 * v131, v131, v130, v129, v70);
  // pto: %rms_2d_inline1751__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v134 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v132, v133);
  // pto: %hc_base_2d_inline1925__ssa_v0_view
  const int64_t v135 = 1;
  // pto: %hc_base_2d_inline1925__ssa_v0_view
  const int64_t v136 = 1;
  // pto: %hc_base_2d_inline1925__ssa_v0_view
  const int64_t v137 = 1;
  // pto: %hc_base_2d_inline1925__ssa_v0_view
  int64_t v138 = v70 * v71;
  // pto: %hc_base_2d_inline1925__ssa_v0_view
  int64_t v139 = v137 * v138;
  // pto: %hc_base_2d_inline1925__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v140 = pto::Shape<1, 1, 1, -1, -1>(v135, v136, v137, v70, v71);
  // pto: %hc_base_2d_inline1925__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v141 = pto::Stride<-1, -1, -1, -1, -1>(v136 * v139, v139, v138, v71, v70);
  // pto: %hc_base_2d_inline1925__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v142 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v8, v140, v141);
  // pto: %comb_inline1738__ssa_v0_view
  const int64_t v143 = 1;
  // pto: %comb_inline1738__ssa_v0_view
  const int64_t v144 = 1;
  // pto: %comb_inline1738__ssa_v0_view
  const int64_t v145 = 1;
  // pto: %comb_inline1738__ssa_v0_view
  int64_t v146 = (int64_t) v30;
  // pto: %comb_inline1738__ssa_v0_view
  int64_t v147 = v146 * v75;
  // pto: %comb_inline1738__ssa_v0_view
  int64_t v148 = v145 * v147;
  // pto: %comb_inline1738__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v149 = pto::Shape<1, 1, 1, -1, -1>(v143, v144, v145, v146, v75);
  // pto: %comb_inline1738__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v150 = pto::Stride<-1, -1, -1, -1, -1>(v144 * v148, v148, v147, v75, v70);
  // pto: %comb_inline1738__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v151 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v9, v149, v150);
  // pto: %hc_scale_2d_inline1730__ssa_v0_view
  const int64_t v152 = 1;
  // pto: %hc_scale_2d_inline1730__ssa_v0_view
  const int64_t v153 = 1;
  // pto: %hc_scale_2d_inline1730__ssa_v0_view
  const int64_t v154 = 1;
  // pto: %hc_scale_2d_inline1730__ssa_v0_view
  int64_t v155 = v70 * v73;
  // pto: %hc_scale_2d_inline1730__ssa_v0_view
  int64_t v156 = v154 * v155;
  // pto: %hc_scale_2d_inline1730__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v157 = pto::Shape<1, 1, 1, -1, -1>(v152, v153, v154, v70, v73);
  // pto: %hc_scale_2d_inline1730__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v158 = pto::Stride<-1, -1, -1, -1, -1>(v153 * v156, v156, v155, v73, v70);
  // pto: %hc_scale_2d_inline1730__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v159 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v10, v157, v158);
  // pto: %245
  int64_t v160 = (int64_t) ((uint64_t) v30 * (uint64_t) v82);
  // pto: %post_flat_inline1702__ssa_v0_view
  const int64_t v161 = 1;
  // pto: %post_flat_inline1702__ssa_v0_view
  const int64_t v162 = 1;
  // pto: %post_flat_inline1702__ssa_v0_view
  const int64_t v163 = 1;
  // pto: %post_flat_inline1702__ssa_v0_view
  int64_t v164 = v70 * v160;
  // pto: %post_flat_inline1702__ssa_v0_view
  int64_t v165 = v163 * v164;
  // pto: %post_flat_inline1702__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v166 = pto::Shape<1, 1, 1, -1, -1>(v161, v162, v163, v70, v160);
  // pto: %post_flat_inline1702__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v167 = pto::Stride<-1, -1, -1, -1, -1>(v162 * v165, v165, v164, v160, v70);
  // pto: %post_flat_inline1702__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v168 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v11, v166, v167);
  // pto: %mixed_view_inline1804__ssa_v0_view
  const int64_t v169 = 1;
  // pto: %mixed_view_inline1804__ssa_v0_view
  const int64_t v170 = 1;
  // pto: %mixed_view_inline1804__ssa_v0_view
  const int64_t v171 = 1;
  // pto: %mixed_view_inline1804__ssa_v0_view
  int64_t v172 = (int64_t) v30;
  // pto: %mixed_view_inline1804__ssa_v0_view
  int64_t v173 = v172 * v76;
  // pto: %mixed_view_inline1804__ssa_v0_view
  int64_t v174 = v171 * v173;
  // pto: %mixed_view_inline1804__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v175 = pto::Shape<1, 1, 1, -1, -1>(v169, v170, v171, v172, v76);
  // pto: %mixed_view_inline1804__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v176 = pto::Stride<-1, -1, -1, -1, -1>(v170 * v174, v174, v173, v76, v70);
  // pto: %mixed_view_inline1804__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v177 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v12, v175, v176);
  const int32_t v178 = 0;
  auto v179 = TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>(v13, v178, v77);
  // pto: %worker_inline1650__ssa_v0
  int64_t v180 = (int64_t) v31;
  // pto: %linear_workers_inline1748__ssa_v0_idx
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  set_flag(PIPE_S, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
  for (int64_t v181 = v180; v181 < v14; v181 += (int64_t) v15) {
    // pto: %46, %47
    ;
    int64_t v182 = (int64_t) ((uint64_t) (v181 / v74) * (uint64_t) v78);
    // pto: %48
    ;
    int64_t v183 = v181 % v74;
    // pto: %49
    ;
    int64_t v184 = (int64_t) ((uint64_t) v183 * (uint64_t) v79);
    // pto: %mid_stream_rms_total_inline1672__tile
    ;
    Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v185 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v69);
    // pto: %mid_stream_rms_total_inline1672__tile
    ;
    uint64_t v186 = (uint64_t) v34;
    TASSIGN(v185, v186);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TEXPANDS(v185, v80);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    for (int64_t v187 = v81; v187 < v82; v187 += v83) {
      // pto: %52
      ;
      int64_t v188 = (int64_t) ((uint64_t) v187 * (uint64_t) v84);
      // pto: %53
      ;
      int64_t v189 = (int64_t) ((uint64_t) v184 + (uint64_t) v188);
      // pto: %55, %lane_inline1799__ssa_v0, %54
      ;
      int64_t v190 = (int64_t) ((uint64_t) v182 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) v33) * (uint64_t) v69));
      // pto: %56
      ;
      int64_t v191 = (int64_t) ((uint64_t) v17 - (uint64_t) v190);
      // pto: %57
      ;
      int64_t v192 = v191 < v69 ? v191 : v69;
      // pto: %58
      ;
      int64_t v193 = v192 < v81 ? v81 : v192;
      // pto: %61, %60
      ;
      int64_t v194 = (int64_t) ((uint64_t) v184 + (uint64_t) ((int64_t) (uint64_t) v188 + (uint64_t) v84));
      // pto: %mid_stream_source_inline1744__tile
      ;
      Tile<TileType::Vec, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v195 = Tile<TileType::Vec, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v193, v84);
      // pto: %mid_stream_source_inline1744__tile
      ;
      uint64_t v196 = (uint64_t) v35;
      TASSIGN(v195, v196);
      // pto: %68
      ;
      int64_t v197 = v190 < v81 ? v81 : v190;
      // pto: %69
      ;
      int64_t v198 = v189 < v81 ? v81 : v189;
      // pto: %x_flat_inline1654__ssa_v0_pview
      ;
      const int64_t v199 = 0;
      // pto: %x_flat_inline1654__ssa_v0_pview
      ;
      __gm__ bfloat16_t* v200 = PTOAS__GLOBAL_TENSOR_DATA(v116);
      // pto: %x_flat_inline1654__ssa_v0_pview
      ;
      const int64_t v201 = 1;
      // pto: %x_flat_inline1654__ssa_v0_pview
      ;
      const int64_t v202 = 1;
      // pto: %x_flat_inline1654__ssa_v0_pview
      ;
      const int64_t v203 = 1;
      // pto: %x_flat_inline1654__ssa_v0_pview
      ;
      int64_t v204 = v193 * v72;
      // pto: %x_flat_inline1654__ssa_v0_pview
      ;
      int64_t v205 = v203 * v204;
      // pto: %x_flat_inline1654__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, -1> v206 = pto::Shape<1, 1, 1, -1, -1>(v201, v202, v203, v193, v84);
      // pto: %x_flat_inline1654__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v207 = pto::Stride<-1, -1, -1, -1, -1>(v202 * v205, v205, v204, v72, v70);
      // pto: %x_flat_inline1654__ssa_v0_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v208 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v200 + (v199 + v197 * v72 + v198 * v70), v206, v207);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v195, v208);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %0
      ;
      Tile<TileType::Vec, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v209 = Tile<TileType::Vec, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v193, v84);
      // pto: %0
      ;
      uint64_t v210 = (uint64_t) v36;
      TASSIGN(v209, v210);
      // pto: %71
      ;
      int64_t v211 = v194 < v81 ? v81 : v194;
      // pto: %72
      ;
      const int64_t v212 = 0;
      // pto: %72
      ;
      __gm__ bfloat16_t* v213 = PTOAS__GLOBAL_TENSOR_DATA(v116);
      // pto: %72
      ;
      const int64_t v214 = 1;
      // pto: %72
      ;
      const int64_t v215 = 1;
      // pto: %72
      ;
      const int64_t v216 = 1;
      // pto: %72
      ;
      int64_t v217 = v193 * v72;
      // pto: %72
      ;
      int64_t v218 = v216 * v217;
      // pto: %72
      ;
      pto::Shape<1, 1, 1, -1, -1> v219 = pto::Shape<1, 1, 1, -1, -1>(v214, v215, v216, v193, v84);
      // pto: %72
      ;
      pto::Stride<-1, -1, -1, -1, -1> v220 = pto::Stride<-1, -1, -1, -1, -1>(v215 * v218, v218, v217, v72, v70);
      // pto: %72
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v221 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v213 + (v212 + v197 * v72 + v211 * v70), v219, v220);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      TLOAD(v209, v221);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %mid_stream_widened_inline1675__tile
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v222 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v193, v84);
      // pto: %mid_stream_widened_inline1675__tile
      ;
      uint64_t v223 = (uint64_t) v37;
      TASSIGN(v222, v223);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      RoundMode v224 = RoundMode::CAST_ROUND;
      SaturationMode v225 = SaturationMode::ON;
      TCVT(v222, v195, v224, v225);
      v222.SetValidShape(v193, v84);
      // pto: %mid_stream_clean_inline1716__tile
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v226 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v69, v84);
      // pto: %mid_stream_clean_inline1716__tile
      ;
      uint64_t v227 = (uint64_t) v37;
      TASSIGN(v226, v227);
      pipe_barrier(PIPE_V);
      TFILLPAD<pto::TFillPadMode::InPlace>(v226, v222);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      v226.SetValidShape(v69, v84);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TPUSH<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v179, v226);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      // pto: %1
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v228 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v69, v84);
      // pto: %1
      ;
      uint64_t v229 = (uint64_t) v37;
      TASSIGN(v228, v229);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      TMUL(v228, v226, v226);
      // pto: %tmp_tile
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v230 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v84);
      // pto: %tmp_tile
      ;
      uint64_t v231 = (uint64_t) v35;
      TASSIGN(v230, v231);
      // pto: %2
      ;
      Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v232 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v70);
      // pto: %2
      ;
      uint64_t v233 = (uint64_t) v38;
      TASSIGN(v232, v233);
      pipe_barrier(PIPE_V);
      TROWSUM(v232, v228, v230);
      // pto: %mid_stream_sq_inline1929__tile
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v234 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v69);
      // pto: %mid_stream_sq_inline1929__tile
      ;
      uint64_t v235 = (uint64_t) v38;
      TASSIGN(v234, v235);
      // pto: %3
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v236 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v69);
      // pto: %3
      ;
      uint64_t v237 = (uint64_t) v38;
      TASSIGN(v236, v237);
      pipe_barrier(PIPE_V);
      TADD(v236, v185, v234);
      // pto: %4
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v238 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v193, v84);
      // pto: %4
      ;
      uint64_t v239 = (uint64_t) v37;
      TASSIGN(v238, v239);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      RoundMode v240 = RoundMode::CAST_ROUND;
      SaturationMode v241 = SaturationMode::ON;
      TCVT(v238, v209, v240, v241);
      v238.SetValidShape(v193, v84);
      // pto: %6
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v242 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v69, v84);
      // pto: %6
      ;
      uint64_t v243 = (uint64_t) v37;
      TASSIGN(v242, v243);
      pipe_barrier(PIPE_V);
      TFILLPAD<pto::TFillPadMode::InPlace>(v242, v238);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      v242.SetValidShape(v69, v84);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      TPUSH<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v179, v242);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      // pto: %8
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v244 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v69, v84);
      // pto: %8
      ;
      uint64_t v245 = (uint64_t) v37;
      TASSIGN(v244, v245);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      TMUL(v244, v242, v242);
      // pto: %9
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v246 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v84);
      // pto: %9
      ;
      uint64_t v247 = (uint64_t) v35;
      TASSIGN(v246, v247);
      // pto: %10
      ;
      Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v248 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v70);
      // pto: %10
      ;
      uint64_t v249 = (uint64_t) v36;
      TASSIGN(v248, v249);
      pipe_barrier(PIPE_V);
      TROWSUM(v248, v244, v246);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      // pto: %11
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v250 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v69);
      // pto: %11
      ;
      uint64_t v251 = (uint64_t) v36;
      TASSIGN(v250, v251);
      // pto: %12
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v252 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v69);
      // pto: %12
      ;
      uint64_t v253 = (uint64_t) v34;
      TASSIGN(v252, v253);
      pipe_barrier(PIPE_V);
      TADD(v252, v236, v250);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    };
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    // pto: %50, %51, %74, %rms_lane_inline1765__ssa_v0, %73
    ;
    int64_t v254 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v183 * (uint64_t) v16) + (uint64_t) v182) + (uint64_t) ((int64_t) (uint64_t) ((int64_t) v33) * (uint64_t) v69));
    // pto: %13
    ;
    Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v255 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v70);
    // pto: %13
    ;
    uint64_t v256 = (uint64_t) v34;
    TASSIGN(v255, v256);
    // pto: %75
    ;
    int64_t v257 = v254 < v81 ? v81 : v254;
    // pto: %rms_partials_inline1725__iter_v1_pview
    ;
    const int64_t v258 = 0;
    // pto: %rms_partials_inline1725__iter_v1_pview
    ;
    __gm__ float* v259 = PTOAS__GLOBAL_TENSOR_DATA(v107);
    // pto: %rms_partials_inline1725__iter_v1_pview
    ;
    const int64_t v260 = 1;
    // pto: %rms_partials_inline1725__iter_v1_pview
    ;
    const int64_t v261 = 1;
    // pto: %rms_partials_inline1725__iter_v1_pview
    ;
    const int64_t v262 = 1;
    // pto: %rms_partials_inline1725__iter_v1_pview
    ;
    int64_t v263 = v69 * v70;
    // pto: %rms_partials_inline1725__iter_v1_pview
    ;
    int64_t v264 = v262 * v263;
    // pto: %rms_partials_inline1725__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 32, 1> v265 = pto::Shape<1, 1, 1, 32, 1>(v260, v261, v262, v69, v70);
    // pto: %rms_partials_inline1725__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v266 = pto::Stride<-1, -1, -1, -1, -1>(v261 * v264, v264, v263, v70, v99);
    // pto: %rms_partials_inline1725__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v267 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v259 + (v258 + v257 * v70 + v81 * v99), v265, v266);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    TSTORE(v267, v255);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  pipe_barrier(PIPE_ALL);
  dsb(DSB_DDR);
  SYNCALL<SyncCoreType::Mix>();
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  // pto: %76, %77, %tail_lane_inline1703__ssa_v0
  int64_t v268 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v180 * (uint64_t) v83) + (uint64_t) ((int64_t) v33));
  // pto: %78
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  if (v268 < v18) {
    // pto: %80
    ;
    int64_t v269 = v268 < v20 ? v268 : v20;
    // pto: %79, %81
    ;
    int64_t v270 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v268 * (uint64_t) v19) + (uint64_t) v269);
    // pto: %83
    ;
    int64_t v271 = (int64_t) ((uint64_t) v20 - (uint64_t) v268);
    // pto: %84
    ;
    int64_t v272 = v271 < v81 ? v81 : v271;
    // pto: %85
    ;
    int64_t v273 = v272 < v70 ? v272 : v70;
    // pto: %82, %86
    ;
    int64_t v274 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v270 + (uint64_t) v19) + (uint64_t) v273);
    for (int64_t v275 = v270; v275 < v274; v275 += v74) {
      // pto: %scale2_inline1756__tile
      ;
      float v276 = (v5)[v83];
      // pto: %87
      ;
      int64_t v277 = (int64_t) ((uint64_t) v274 - (uint64_t) v275);
      // pto: %88
      ;
      int64_t v278 = v277 < v74 ? v277 : v74;
      // pto: %89
      ;
      int64_t v279 = (int64_t) ((uint64_t) v278 * (uint64_t) v69);
      // pto: %c_parts_load_inline1838__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v280 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v279);
      // pto: %c_parts_load_inline1838__ssa_v0
      ;
      uint64_t v281 = (uint64_t) v37;
      TASSIGN(v280, v281);
      // pto: %90
      ;
      int64_t v282 = (int64_t) ((uint64_t) v275 * (uint64_t) v69);
      // pto: %91
      ;
      int64_t v283 = v282 < v81 ? v81 : v282;
      // pto: %parts_2d_inline1739__ssa_v0_pview
      ;
      const int64_t v284 = 0;
      // pto: %parts_2d_inline1739__ssa_v0_pview
      ;
      __gm__ float* v285 = PTOAS__GLOBAL_TENSOR_DATA(v125);
      // pto: %parts_2d_inline1739__ssa_v0_pview
      ;
      const int64_t v286 = 1;
      // pto: %parts_2d_inline1739__ssa_v0_pview
      ;
      const int64_t v287 = 1;
      // pto: %parts_2d_inline1739__ssa_v0_pview
      ;
      const int64_t v288 = 1;
      // pto: %parts_2d_inline1739__ssa_v0_pview
      ;
      int64_t v289 = v74 * v117;
      // pto: %parts_2d_inline1739__ssa_v0_pview
      ;
      int64_t v290 = v288 * v289;
      // pto: %parts_2d_inline1739__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, -1> v291 = pto::Shape<1, 1, 1, 8, -1>(v286, v287, v288, v74, v279);
      // pto: %parts_2d_inline1739__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v292 = pto::Stride<-1, -1, -1, -1, -1>(v287 * v290, v290, v289, v117, v70);
      // pto: %parts_2d_inline1739__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v293 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v285 + (v284 + v81 * v117 + v283 * v70), v291, v292);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      TLOAD(v280, v293);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %t__tmp_v401
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v294 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v74, v85);
      // pto: %t__tmp_v401
      ;
      uint64_t v295 = (uint64_t) v37;
      TASSIGN(v294, v295);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      TFILLPAD<pto::TFillPadMode::InPlace>(v294, v280);
      v294.SetValidShape(v74, v85);
      // pto: %c_parts_tmp_inline1778__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v296 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v85);
      // pto: %c_parts_tmp_inline1778__ssa_v0
      ;
      uint64_t v297 = (uint64_t) v39;
      TASSIGN(v296, v297);
      // pto: %t__tmp_v402
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v298 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v85);
      // pto: %t__tmp_v402
      ;
      uint64_t v299 = (uint64_t) v37;
      TASSIGN(v298, v299);
      // pto: %colsum_src_view
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, 8, 256, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v300;
      TRESHAPE(v300, v294);
      // pto: %colsum_tmp_view
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, 8, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v301;
      TRESHAPE(v301, v296);
      // pto: %colsum_dst_view
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v302;
      TRESHAPE(v302, v298);
      pipe_barrier(PIPE_V);
      const bool v303 = true;
      TCOLSUM(v302, v300, v301, v303);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      // pto: %c_total_inline1652__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v304 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v69);
      // pto: %c_total_inline1652__ssa_v0
      ;
      uint64_t v305 = (uint64_t) v37;
      TASSIGN(v304, v305);
      // pto: %c_sq_load_inline1819__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v306 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v278);
      // pto: %c_sq_load_inline1819__ssa_v0
      ;
      uint64_t v307 = (uint64_t) v40;
      TASSIGN(v306, v307);
      // pto: %93
      ;
      int64_t v308 = v275 < v81 ? v81 : v275;
      // pto: %rms_2d_inline1751__ssa_v0_pview
      ;
      int64_t v309 = (int64_t) v29;
      // pto: %rms_2d_inline1751__ssa_v0_pview
      ;
      const int64_t v310 = 0;
      // pto: %rms_2d_inline1751__ssa_v0_pview
      ;
      __gm__ float* v311 = PTOAS__GLOBAL_TENSOR_DATA(v134);
      // pto: %rms_2d_inline1751__ssa_v0_pview
      ;
      const int64_t v312 = 1;
      // pto: %rms_2d_inline1751__ssa_v0_pview
      ;
      const int64_t v313 = 1;
      // pto: %rms_2d_inline1751__ssa_v0_pview
      ;
      const int64_t v314 = 1;
      // pto: %rms_2d_inline1751__ssa_v0_pview
      ;
      int64_t v315 = v74 * v309;
      // pto: %rms_2d_inline1751__ssa_v0_pview
      ;
      int64_t v316 = v314 * v315;
      // pto: %rms_2d_inline1751__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, -1> v317 = pto::Shape<1, 1, 1, 8, -1>(v312, v313, v314, v74, v278);
      // pto: %rms_2d_inline1751__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v318 = pto::Stride<-1, -1, -1, -1, -1>(v313 * v316, v316, v315, v309, v70);
      // pto: %rms_2d_inline1751__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v319 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v311 + (v310 + v81 * v309 + v308 * v70), v317, v318);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      TLOAD(v306, v319);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      // pto: %t__tmp_v403
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v320 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v74, v74);
      // pto: %t__tmp_v403
      ;
      uint64_t v321 = (uint64_t) v40;
      TASSIGN(v320, v321);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      TFILLPAD<pto::TFillPadMode::InPlace>(v320, v306);
      v320.SetValidShape(v74, v74);
      // pto: %c_sq_tmp_inline1759__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v322 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v74);
      // pto: %c_sq_tmp_inline1759__ssa_v0
      ;
      uint64_t v323 = (uint64_t) v41;
      TASSIGN(v322, v323);
      // pto: %c_rms_tmp_inline1923__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v324 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v74);
      // pto: %c_rms_tmp_inline1923__ssa_v0
      ;
      uint64_t v325 = (uint64_t) v42;
      TASSIGN(v324, v325);
      // pto: %t__tmp_v404
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v326 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v74);
      // pto: %t__tmp_v404
      ;
      uint64_t v327 = (uint64_t) v40;
      TASSIGN(v326, v327);
      // pto: %94
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v328;
      TRESHAPE(v328, v320);
      // pto: %95
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v329;
      TRESHAPE(v329, v322);
      // pto: %96
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v330;
      TRESHAPE(v330, v326);
      pipe_barrier(PIPE_V);
      const bool v331 = true;
      TCOLSUM(v330, v328, v329, v331);
      // pto: %t__tmp_v405
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v332 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v74);
      // pto: %t__tmp_v405
      ;
      uint64_t v333 = (uint64_t) v40;
      TASSIGN(v332, v333);
      pipe_barrier(PIPE_V);
      TMULS(v332, v326, v86);
      // pto: %t__tmp_v406
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v334 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v74);
      // pto: %t__tmp_v406
      ;
      uint64_t v335 = (uint64_t) v40;
      TASSIGN(v334, v335);
      pipe_barrier(PIPE_V);
      TADDS(v334, v332, v87);
      // pto: %t__tmp_v407
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v336 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v74);
      // pto: %t__tmp_v407
      ;
      uint64_t v337 = (uint64_t) v43;
      TASSIGN(v336, v337);
      pipe_barrier(PIPE_V);
      TRSQRT(v336, v334, v324);
      // pto: %c_inv_inline1678__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v338 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v70);
      // pto: %c_inv_inline1678__ssa_v0
      ;
      uint64_t v339 = (uint64_t) v43;
      TASSIGN(v338, v339);
      // pto: %comb_inv_inline1807__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v340 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v70);
      // pto: %comb_inv_inline1807__ssa_v0
      ;
      uint64_t v341 = (uint64_t) v43;
      TASSIGN(v340, v341);
      // pto: %t__tmp_v408
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v342 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v75);
      // pto: %t__tmp_v408
      ;
      uint64_t v343 = (uint64_t) v44;
      TASSIGN(v342, v343);
      // pto: %slice_view
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 16, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v344;
      // pto: %slice_view
      ;
      uint64_t v345 = (uint64_t) v44;
      TASSIGN(v344, v345);
      // pto: %raw_inline1930__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v346 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v75);
      // pto: %raw_inline1930__ssa_v0
      ;
      uint64_t v347 = (uint64_t) v37;
      TASSIGN(v346, v347);
      TADDS(v346, v344, v80);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      // pto: %base_inline1768__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v348 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v75);
      // pto: %base_inline1768__ssa_v0
      ;
      uint64_t v349 = (uint64_t) v45;
      TASSIGN(v348, v349);
      // pto: %hc_base_2d_inline1925__ssa_v0_pview
      ;
      __gm__ float* v350 = PTOAS__GLOBAL_TENSOR_DATA(v142);
      // pto: %hc_base_2d_inline1925__ssa_v0_pview
      ;
      unsigned v351 = 8;
      // pto: %hc_base_2d_inline1925__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 16> v352 = pto::Shape<1, 1, 1, 1, 16>();
      // pto: %hc_base_2d_inline1925__ssa_v0_pview
      ;
      pto::Stride<24, 24, 24, 24, 1> v353 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %hc_base_2d_inline1925__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v354 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v350 + v351, v352, v353);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      TLOAD(v348, v354);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      // pto: %t__tmp_v409
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v355 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v75);
      // pto: %t__tmp_v409
      ;
      uint64_t v356 = (uint64_t) v37;
      TASSIGN(v355, v356);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v355, v346, v340);
      // pto: %t__tmp_v410
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v357 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v75);
      // pto: %t__tmp_v410
      ;
      uint64_t v358 = (uint64_t) v37;
      TASSIGN(v357, v358);
      pipe_barrier(PIPE_V);
      TMULS(v357, v355, v276);
      // pto: %logits_inline1810__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v359 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v75);
      // pto: %logits_inline1810__ssa_v0
      ;
      uint64_t v360 = (uint64_t) v46;
      TASSIGN(v359, v360);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      TCOLEXPANDADD(v359, v357, v348);
      // pto: %t__tmp_v411
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v361 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v75);
      // pto: %t__tmp_v411
      ;
      uint64_t v362 = (uint64_t) v46;
      TASSIGN(v361, v362);
      // pto: %regular_logits_inline1662__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v363 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %regular_logits_inline1662__ssa_v0
      ;
      uint64_t v364 = (uint64_t) v46;
      TASSIGN(v363, v364);
      // pto: %ids_inline1755__ci_tmp_v0
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v365 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v89);
      // pto: %ids_inline1755__ci_tmp_v0
      ;
      uint64_t v366 = (uint64_t) v37;
      TASSIGN(v365, v366);
      // pto: %ids_inline1755__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v367 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %ids_inline1755__ssa_v1
      ;
      uint64_t v368 = (uint64_t) v41;
      TASSIGN(v367, v368);
      // pto: %ci_tmp_view
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v369;
      TRESHAPE(v369, v365);
      // pto: %ci_dst_view
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v370;
      TRESHAPE(v370, v367);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v370, v77, v369);
      // pto: %tmp_inline1813__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v371 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %tmp_inline1813__ssa_v1
      ;
      uint64_t v372 = (uint64_t) v37;
      TASSIGN(v371, v372);
      // pto: %ids16_inline1825__ssa_v1
      ;
      Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v373 = Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v85);
      // pto: %ids16_inline1825__ssa_v1
      ;
      uint64_t v374 = (uint64_t) v41;
      TASSIGN(v373, v374);
      // pto: %xor_tmp16_inline1826__ssa_v1
      ;
      Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v375 = Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v85);
      // pto: %xor_tmp16_inline1826__ssa_v1
      ;
      uint64_t v376 = (uint64_t) v47;
      TASSIGN(v375, v376);
      // pto: %t__tmp_v412
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v377 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %t__tmp_v412
      ;
      uint64_t v378 = (uint64_t) v48;
      TASSIGN(v377, v378);
      TEXPANDS(v377, v90);
      // pto: %ri1_mask_inline1770__ssa_v1
      ;
      Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v379 = Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v85);
      // pto: %ri1_mask_inline1770__ssa_v1
      ;
      uint64_t v380 = (uint64_t) v48;
      TASSIGN(v379, v380);
      // pto: %t__tmp_v413
      ;
      Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v381 = Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v85);
      // pto: %t__tmp_v413
      ;
      uint64_t v382 = (uint64_t) v49;
      TASSIGN(v381, v382);
      pipe_barrier(PIPE_V);
      TXOR(v381, v373, v379, v375);
      // pto: %ri1_inline1833__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v383 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %ri1_inline1833__ssa_v1
      ;
      uint64_t v384 = (uint64_t) v49;
      TASSIGN(v383, v384);
      // pto: %t__tmp_v414
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v385 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %t__tmp_v414
      ;
      uint64_t v386 = (uint64_t) v48;
      TASSIGN(v385, v386);
      pipe_barrier(PIPE_V);
      TEXPANDS(v385, v91);
      // pto: %ri2_mask_inline1811__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v387 = Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v85);
      // pto: %ri2_mask_inline1811__ssa_v0
      ;
      uint64_t v388 = (uint64_t) v48;
      TASSIGN(v387, v388);
      // pto: %t__tmp_v415
      ;
      Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v389 = Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v85);
      // pto: %t__tmp_v415
      ;
      uint64_t v390 = (uint64_t) v50;
      TASSIGN(v389, v390);
      pipe_barrier(PIPE_V);
      TXOR(v389, v373, v387, v375);
      // pto: %ri2_inline1868__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v391 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %ri2_inline1868__ssa_v0
      ;
      uint64_t v392 = (uint64_t) v50;
      TASSIGN(v391, v392);
      // pto: %t__tmp_v416
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v393 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %t__tmp_v416
      ;
      uint64_t v394 = (uint64_t) v48;
      TASSIGN(v393, v394);
      pipe_barrier(PIPE_V);
      TEXPANDS(v393, v92);
      // pto: %ci1_mask_inline1850__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v395 = Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v85);
      // pto: %ci1_mask_inline1850__ssa_v0
      ;
      uint64_t v396 = (uint64_t) v48;
      TASSIGN(v395, v396);
      // pto: %t__tmp_v417
      ;
      Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v397 = Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v85);
      // pto: %t__tmp_v417
      ;
      uint64_t v398 = (uint64_t) v51;
      TASSIGN(v397, v398);
      pipe_barrier(PIPE_V);
      TXOR(v397, v373, v395, v375);
      // pto: %ci1_inline1869__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v399 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %ci1_inline1869__ssa_v0
      ;
      uint64_t v400 = (uint64_t) v51;
      TASSIGN(v399, v400);
      // pto: %t__tmp_v418
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v401 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %t__tmp_v418
      ;
      uint64_t v402 = (uint64_t) v48;
      TASSIGN(v401, v402);
      pipe_barrier(PIPE_V);
      TEXPANDS(v401, v93);
      // pto: %ci2_mask_inline1921__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v403 = Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v85);
      // pto: %ci2_mask_inline1921__ssa_v0
      ;
      uint64_t v404 = (uint64_t) v48;
      TASSIGN(v403, v404);
      // pto: %t__tmp_v419
      ;
      Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v405 = Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v85);
      // pto: %t__tmp_v419
      ;
      uint64_t v406 = (uint64_t) v52;
      TASSIGN(v405, v406);
      pipe_barrier(PIPE_V);
      TXOR(v405, v373, v403, v375);
      // pto: %ci2_inline1803__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v407 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %ci2_inline1803__ssa_v0
      ;
      uint64_t v408 = (uint64_t) v52;
      TASSIGN(v407, v408);
      // pto: %t__tmp_v420
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v409 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %t__tmp_v420
      ;
      uint64_t v410 = (uint64_t) v41;
      TASSIGN(v409, v410);
      pipe_barrier(PIPE_V);
      TGATHER(v409, v363, v383, v371);
      // pto: %max2_inline1852__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v411 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %max2_inline1852__ssa_v1
      ;
      uint64_t v412 = (uint64_t) v41;
      TASSIGN(v411, v412);
      pipe_barrier(PIPE_V);
      TMAX(v411, v363, v409);
      // pto: %t__tmp_v421
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v413 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %t__tmp_v421
      ;
      uint64_t v414 = (uint64_t) v47;
      TASSIGN(v413, v414);
      pipe_barrier(PIPE_V);
      TGATHER(v413, v411, v391, v371);
      // pto: %max4_inline1856__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v415 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %max4_inline1856__ssa_v1
      ;
      uint64_t v416 = (uint64_t) v41;
      TASSIGN(v415, v416);
      pipe_barrier(PIPE_V);
      TMAX(v415, v411, v413);
      // pto: %t__tmp_v422
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v417 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %t__tmp_v422
      ;
      uint64_t v418 = (uint64_t) v45;
      TASSIGN(v417, v418);
      pipe_barrier(PIPE_V);
      TSUB(v417, v363, v415);
      // pto: %exps_inline1771__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v419 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %exps_inline1771__ssa_v1
      ;
      uint64_t v420 = (uint64_t) v45;
      TASSIGN(v419, v420);
      pipe_barrier(PIPE_V);
      TEXP(v419, v417);
      // pto: %t__tmp_v423
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v421 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %t__tmp_v423
      ;
      uint64_t v422 = (uint64_t) v40;
      TASSIGN(v421, v422);
      pipe_barrier(PIPE_V);
      TGATHER(v421, v419, v383, v371);
      // pto: %sum2_inline1805__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v423 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %sum2_inline1805__ssa_v1
      ;
      uint64_t v424 = (uint64_t) v40;
      TASSIGN(v423, v424);
      pipe_barrier(PIPE_V);
      TADD(v423, v419, v421);
      // pto: %t__tmp_v424
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v425 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %t__tmp_v424
      ;
      uint64_t v426 = (uint64_t) v42;
      TASSIGN(v425, v426);
      pipe_barrier(PIPE_V);
      TGATHER(v425, v423, v391, v371);
      // pto: %sum4_inline1859__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v427 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %sum4_inline1859__ssa_v1
      ;
      uint64_t v428 = (uint64_t) v40;
      TASSIGN(v427, v428);
      pipe_barrier(PIPE_V);
      TADD(v427, v423, v425);
      // pto: %t__tmp_v425
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v429 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %t__tmp_v425
      ;
      uint64_t v430 = (uint64_t) v45;
      TASSIGN(v429, v430);
      pipe_barrier(PIPE_V);
      TDIV(v429, v419, v427);
      // pto: %probs_inline1795__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v431 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %probs_inline1795__ssa_v1
      ;
      uint64_t v432 = (uint64_t) v45;
      TASSIGN(v431, v432);
      pipe_barrier(PIPE_V);
      TADDS(v431, v429, v87);
      // pto: %t__tmp_v426
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v433 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %t__tmp_v426
      ;
      uint64_t v434 = (uint64_t) v40;
      TASSIGN(v433, v434);
      pipe_barrier(PIPE_V);
      TGATHER(v433, v431, v399, v371);
      // pto: %col2_inline1729__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v435 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %col2_inline1729__ssa_v4
      ;
      uint64_t v436 = (uint64_t) v40;
      TASSIGN(v435, v436);
      pipe_barrier(PIPE_V);
      TADD(v435, v431, v433);
      // pto: %t__tmp_v427
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v437 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %t__tmp_v427
      ;
      uint64_t v438 = (uint64_t) v42;
      TASSIGN(v437, v438);
      pipe_barrier(PIPE_V);
      TGATHER(v437, v435, v407, v371);
      // pto: %t__tmp_v428
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v439 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %t__tmp_v428
      ;
      uint64_t v440 = (uint64_t) v40;
      TASSIGN(v439, v440);
      pipe_barrier(PIPE_V);
      TADD(v439, v435, v437);
      // pto: %col4_inline1690__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v441 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %col4_inline1690__ssa_v4
      ;
      uint64_t v442 = (uint64_t) v40;
      TASSIGN(v441, v442);
      pipe_barrier(PIPE_V);
      TADDS(v441, v439, v87);
      // pto: %cur_inline1718__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v443 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
      // pto: %cur_inline1718__ssa_v4
      ;
      uint64_t v444 = (uint64_t) v45;
      TASSIGN(v443, v444);
      pipe_barrier(PIPE_V);
      TDIV(v443, v431, v441);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      for (int64_t v445 = v81; v445 < v94; v445 += v70) {
        // pto: %t__tmp_v429
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v446 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
        // pto: %t__tmp_v429
        ;
        uint64_t v447 = (uint64_t) v40;
        TASSIGN(v446, v447);
        pipe_barrier(PIPE_V);
        TGATHER(v446, v443, v383, v371);
        // pto: %row2_inline1863__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v448 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
        // pto: %row2_inline1863__ssa_v1
        ;
        uint64_t v449 = (uint64_t) v40;
        TASSIGN(v448, v449);
        pipe_barrier(PIPE_V);
        TADD(v448, v443, v446);
        // pto: %t__tmp_v430
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v450 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
        // pto: %t__tmp_v430
        ;
        uint64_t v451 = (uint64_t) v42;
        TASSIGN(v450, v451);
        pipe_barrier(PIPE_V);
        TGATHER(v450, v448, v391, v371);
        // pto: %t__tmp_v431
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v452 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
        // pto: %t__tmp_v431
        ;
        uint64_t v453 = (uint64_t) v40;
        TASSIGN(v452, v453);
        pipe_barrier(PIPE_V);
        TADD(v452, v448, v450);
        // pto: %row4_inline1691__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v454 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
        // pto: %row4_inline1691__ssa_v1
        ;
        uint64_t v455 = (uint64_t) v40;
        TASSIGN(v454, v455);
        pipe_barrier(PIPE_V);
        TADDS(v454, v452, v87);
        // pto: %norm_inline1865__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v456 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
        // pto: %norm_inline1865__ssa_v1
        ;
        uint64_t v457 = (uint64_t) v40;
        TASSIGN(v456, v457);
        pipe_barrier(PIPE_V);
        TDIV(v456, v443, v454);
        // pto: %t__tmp_v432
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v458 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
        // pto: %t__tmp_v432
        ;
        uint64_t v459 = (uint64_t) v42;
        TASSIGN(v458, v459);
        pipe_barrier(PIPE_V);
        TGATHER(v458, v456, v399, v371);
        // pto: %col2_inline1729__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v460 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
        // pto: %col2_inline1729__ssa_v7
        ;
        uint64_t v461 = (uint64_t) v42;
        TASSIGN(v460, v461);
        pipe_barrier(PIPE_V);
        TADD(v460, v456, v458);
        // pto: %t__tmp_v433
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v462 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
        // pto: %t__tmp_v433
        ;
        uint64_t v463 = (uint64_t) v53;
        TASSIGN(v462, v463);
        pipe_barrier(PIPE_V);
        TGATHER(v462, v460, v407, v371);
        // pto: %t__tmp_v434
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v464 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
        // pto: %t__tmp_v434
        ;
        uint64_t v465 = (uint64_t) v42;
        TASSIGN(v464, v465);
        pipe_barrier(PIPE_V);
        TADD(v464, v460, v462);
        // pto: %col4_inline1690__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v466 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
        // pto: %col4_inline1690__ssa_v7
        ;
        uint64_t v467 = (uint64_t) v42;
        TASSIGN(v466, v467);
        pipe_barrier(PIPE_V);
        TADDS(v466, v464, v87);
        // pto: %cur_inline1718__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v468 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v88);
        // pto: %cur_inline1718__ssa_v7
        ;
        uint64_t v469 = (uint64_t) v45;
        TASSIGN(v468, v469);
        pipe_barrier(PIPE_V);
        TDIV(v468, v456, v466);
      };
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      // pto: %t__tmp_v435
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v470 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v75);
      // pto: %t__tmp_v435
      ;
      uint64_t v471 = (uint64_t) v45;
      TASSIGN(v470, v471);
      v470.SetValidShape(v278, v75);
      // pto: %comb_inline1738__ssa_v0_pview
      ;
      const int64_t v472 = 0;
      // pto: %comb_inline1738__ssa_v0_pview
      ;
      __gm__ float* v473 = PTOAS__GLOBAL_TENSOR_DATA(v151);
      // pto: %comb_inline1738__ssa_v0_pview
      ;
      const int64_t v474 = 1;
      // pto: %comb_inline1738__ssa_v0_pview
      ;
      const int64_t v475 = 1;
      // pto: %comb_inline1738__ssa_v0_pview
      ;
      const int64_t v476 = 1;
      // pto: %comb_inline1738__ssa_v0_pview
      ;
      int64_t v477 = v278 * v75;
      // pto: %comb_inline1738__ssa_v0_pview
      ;
      int64_t v478 = v476 * v477;
      // pto: %comb_inline1738__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, 16> v479 = pto::Shape<1, 1, 1, -1, 16>(v474, v475, v476, v278, v75);
      // pto: %comb_inline1738__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v480 = pto::Stride<-1, -1, -1, -1, -1>(v475 * v478, v478, v477, v75, v70);
      // pto: %comb_inline1738__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v481 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v473 + (v472 + v308 * v75 + v81 * v70), v479, v480);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      TSTORE(v481, v470);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    };
  } else {
    // pto: %99
    ;
    int64_t v482 = (int64_t) ((uint64_t) v268 - (uint64_t) v18);
    // pto: %101
    ;
    int64_t v483 = v482 < v22 ? v482 : v22;
    // pto: %100, %102, %103
    ;
    int64_t v484 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v482 * (uint64_t) v21) + (uint64_t) v483) * (uint64_t) v83);
    // pto: %104
    ;
    int64_t v485 = (int64_t) ((uint64_t) v22 - (uint64_t) v482);
    // pto: %105
    ;
    int64_t v486 = v485 < v81 ? v81 : v485;
    // pto: %106
    ;
    int64_t v487 = v486 < v70 ? v486 : v70;
    // pto: %109, %107, %108
    ;
    int64_t v488 = (int64_t) ((uint64_t) v484 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v21 + (uint64_t) v487) * (uint64_t) v83));
    // pto: %110
    ;
    // pto: %111
    ;
    // pto: %d_begin_inline1902__phi_v2
    ;
    int64_t v489;
    // pto: %work_begin_inline1924__phi_v3
    ;
    int64_t v490;
    // pto: %work_end_inline1661__phi_v3
    ;
    int64_t v491;
    if (v24 < v25) {
      // pto: %112
      ;
      // pto: %d_worker_inline1860__phi_v2
      ;
      int64_t v492;
      // pto: %d_workers_inline1885__phi_v2
      ;
      int64_t v493;
      // pto: %pair_id_inline1910__phi_v2
      ;
      int64_t v494;
      if (v482 < v26) {
        // pto: %113
        ;
        int64_t v495 = (int64_t) ((uint64_t) v27 + (uint64_t) v70);
        // pto: %114
        ;
        // pto: %116
        ;
        v492 = v482 % v495;
        v493 = v495;
        v494 = v482 / v495;
      } else {
        // pto: %118
        ;
        int64_t v496 = (int64_t) ((uint64_t) v482 - (uint64_t) v26);
        // pto: %120, %119
        ;
        // pto: %121
        ;
        v492 = v496 % v27;
        v493 = v27;
        v494 = (int64_t) ((uint64_t) v28 + (uint64_t) (v496 / v27));
      };
      // pto: %122
      ;
      uint64_t v497 = (uint64_t) v494;
      // pto: %122
      ;
      int64_t v498 = (int64_t) (v497 * (uint64_t) v83);
      // pto: %123
      ;
      int64_t v499 = v23 < v498 ? v23 : v498;
      // pto: %124
      ;
      int64_t v500 = (int64_t) ((uint64_t) v499 + (uint64_t) v83);
      // pto: %125
      ;
      // pto: %126
      ;
      int64_t v501 = v74 / v493;
      // pto: %127
      ;
      int64_t v502 = v74 % v493;
      // pto: %128
      ;
      uint64_t v503 = (uint64_t) v492;
      // pto: %129
      ;
      bool v504 = v492 < v502;
      // pto: %129
      ;
      int64_t v505 = v504 ? v492 : v502;
      // pto: %128, %130
      ;
      v489 = (int64_t) ((uint64_t) ((int64_t) v503 * (uint64_t) v501) + (uint64_t) v505);
      v490 = v499;
      v491 = v23 < v500 ? v23 : v500;
    } else {
      v489 = v81;
      v490 = v484;
      v491 = v23 < v488 ? v23 : v488;
    };
    // pto: %131
    ;
    bool v506 = v25 <= v482;
    // pto: %work_begin_inline1924__phi_v5
    ;
    int64_t v507 = v506 ? v81 : v490;
    // pto: %work_end_inline1661__phi_v5
    ;
    int64_t v508 = v506 ? v81 : v491;
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    for (int64_t v509 = v507; v509 < v508; v509 += v74) {
      // pto: %132
      ;
      int64_t v510 = (int64_t) ((uint64_t) v508 - (uint64_t) v509);
      // pto: %133
      ;
      int64_t v511 = v510 < v74 ? v510 : v74;
      // pto: %134
      ;
      int64_t v512 = (int64_t) ((uint64_t) v511 * (uint64_t) v69);
      // pto: %m_parts_load_inline1908__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v513 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v512);
      // pto: %m_parts_load_inline1908__ssa_v0
      ;
      uint64_t v514 = (uint64_t) v37;
      TASSIGN(v513, v514);
      // pto: %135
      ;
      int64_t v515 = (int64_t) ((uint64_t) v509 * (uint64_t) v69);
      // pto: %136
      ;
      int64_t v516 = v515 < v81 ? v81 : v515;
      // pto: %138
      ;
      const int64_t v517 = 0;
      // pto: %138
      ;
      __gm__ float* v518 = PTOAS__GLOBAL_TENSOR_DATA(v125);
      // pto: %138
      ;
      const int64_t v519 = 1;
      // pto: %138
      ;
      const int64_t v520 = 1;
      // pto: %138
      ;
      const int64_t v521 = 1;
      // pto: %138
      ;
      int64_t v522 = v74 * v117;
      // pto: %138
      ;
      int64_t v523 = v521 * v522;
      // pto: %138
      ;
      pto::Shape<1, 1, 1, 8, -1> v524 = pto::Shape<1, 1, 1, 8, -1>(v519, v520, v521, v74, v512);
      // pto: %138
      ;
      pto::Stride<-1, -1, -1, -1, -1> v525 = pto::Stride<-1, -1, -1, -1, -1>(v520 * v523, v523, v522, v117, v70);
      // pto: %138
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v526 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v518 + (v517 + v81 * v117 + v516 * v70), v524, v525);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      TLOAD(v513, v526);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      // pto: %t__tmp_v436
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v527 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v74, v85);
      // pto: %t__tmp_v436
      ;
      uint64_t v528 = (uint64_t) v37;
      TASSIGN(v527, v528);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      TFILLPAD<pto::TFillPadMode::InPlace>(v527, v513);
      v527.SetValidShape(v74, v85);
      // pto: %m_parts_tmp_inline1915__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v529 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v85);
      // pto: %m_parts_tmp_inline1915__ssa_v0
      ;
      uint64_t v530 = (uint64_t) v39;
      TASSIGN(v529, v530);
      // pto: %t__tmp_v437
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v531 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v85);
      // pto: %t__tmp_v437
      ;
      uint64_t v532 = (uint64_t) v37;
      TASSIGN(v531, v532);
      // pto: %139
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, 8, 256, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v533;
      TRESHAPE(v533, v527);
      // pto: %140
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, 8, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v534;
      TRESHAPE(v534, v529);
      // pto: %141
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v535;
      TRESHAPE(v535, v531);
      pipe_barrier(PIPE_V);
      const bool v536 = true;
      TCOLSUM(v535, v533, v534, v536);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      // pto: %m_total_inline1916__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v537 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v69);
      // pto: %m_total_inline1916__ssa_v0
      ;
      uint64_t v538 = (uint64_t) v37;
      TASSIGN(v537, v538);
      // pto: %m_sq_load_inline1750__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v539 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v511);
      // pto: %m_sq_load_inline1750__ssa_v0
      ;
      uint64_t v540 = (uint64_t) v40;
      TASSIGN(v539, v540);
      // pto: %142
      ;
      int64_t v541 = v509 < v81 ? v81 : v509;
      // pto: %143
      ;
      int64_t v542 = (int64_t) v29;
      // pto: %143
      ;
      const int64_t v543 = 0;
      // pto: %143
      ;
      __gm__ float* v544 = PTOAS__GLOBAL_TENSOR_DATA(v134);
      // pto: %143
      ;
      const int64_t v545 = 1;
      // pto: %143
      ;
      const int64_t v546 = 1;
      // pto: %143
      ;
      const int64_t v547 = 1;
      // pto: %143
      ;
      int64_t v548 = v74 * v542;
      // pto: %143
      ;
      int64_t v549 = v547 * v548;
      // pto: %143
      ;
      pto::Shape<1, 1, 1, 8, -1> v550 = pto::Shape<1, 1, 1, 8, -1>(v545, v546, v547, v74, v511);
      // pto: %143
      ;
      pto::Stride<-1, -1, -1, -1, -1> v551 = pto::Stride<-1, -1, -1, -1, -1>(v546 * v549, v549, v548, v542, v70);
      // pto: %143
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v552 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v544 + (v543 + v81 * v542 + v541 * v70), v550, v551);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      TLOAD(v539, v552);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      // pto: %t__tmp_v438
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v553 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v74, v74);
      // pto: %t__tmp_v438
      ;
      uint64_t v554 = (uint64_t) v40;
      TASSIGN(v553, v554);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      TFILLPAD<pto::TFillPadMode::InPlace>(v553, v539);
      v553.SetValidShape(v74, v74);
      // pto: %m_sq_tmp_inline1835__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v555 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v74);
      // pto: %m_sq_tmp_inline1835__ssa_v0
      ;
      uint64_t v556 = (uint64_t) v41;
      TASSIGN(v555, v556);
      // pto: %m_rms_tmp_inline1918__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v557 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v74);
      // pto: %m_rms_tmp_inline1918__ssa_v0
      ;
      uint64_t v558 = (uint64_t) v42;
      TASSIGN(v557, v558);
      // pto: %t__tmp_v439
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v559 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v74);
      // pto: %t__tmp_v439
      ;
      uint64_t v560 = (uint64_t) v40;
      TASSIGN(v559, v560);
      // pto: %144
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v561;
      TRESHAPE(v561, v553);
      // pto: %145
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v562;
      TRESHAPE(v562, v555);
      // pto: %146
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v563;
      TRESHAPE(v563, v559);
      pipe_barrier(PIPE_V);
      const bool v564 = true;
      TCOLSUM(v563, v561, v562, v564);
      // pto: %t__tmp_v440
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v565 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v74);
      // pto: %t__tmp_v440
      ;
      uint64_t v566 = (uint64_t) v40;
      TASSIGN(v565, v566);
      pipe_barrier(PIPE_V);
      TMULS(v565, v559, v86);
      // pto: %t__tmp_v441
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v567 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v74);
      // pto: %t__tmp_v441
      ;
      uint64_t v568 = (uint64_t) v40;
      TASSIGN(v567, v568);
      pipe_barrier(PIPE_V);
      TADDS(v567, v565, v87);
      // pto: %t__tmp_v442
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v569 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v74);
      // pto: %t__tmp_v442
      ;
      uint64_t v570 = (uint64_t) v47;
      TASSIGN(v569, v570);
      pipe_barrier(PIPE_V);
      TRSQRT(v569, v567, v557);
      // pto: %m_inv_inline1683__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v571 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v70);
      // pto: %m_inv_inline1683__ssa_v0
      ;
      uint64_t v572 = (uint64_t) v47;
      TASSIGN(v571, v572);
      // pto: %gate_base_inline1787__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v573 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v74);
      // pto: %gate_base_inline1787__ssa_v0
      ;
      uint64_t v574 = (uint64_t) v54;
      TASSIGN(v573, v574);
      // pto: %147
      ;
      __gm__ float* v575 = PTOAS__GLOBAL_TENSOR_DATA(v142);
      // pto: %147
      ;
      pto::Shape<1, 1, 1, 1, 8> v576 = pto::Shape<1, 1, 1, 1, 8>();
      // pto: %147
      ;
      pto::Stride<24, 24, 24, 24, 1> v577 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %147
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v578 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v575, v576, v577);
      TLOAD(v573, v578);
      // pto: %gate_scales_inline1919__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v579 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v73);
      // pto: %gate_scales_inline1919__ssa_v0
      ;
      uint64_t v580 = (uint64_t) v55;
      TASSIGN(v579, v580);
      // pto: %hc_scale_2d_inline1730__ssa_v0_pview
      ;
      __gm__ float* v581 = PTOAS__GLOBAL_TENSOR_DATA(v159);
      // pto: %hc_scale_2d_inline1730__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 3> v582 = pto::Shape<1, 1, 1, 1, 3>();
      // pto: %hc_scale_2d_inline1730__ssa_v0_pview
      ;
      pto::Stride<3, 3, 3, 3, 1> v583 = pto::Stride<3, 3, 3, 3, 1>();
      // pto: %hc_scale_2d_inline1730__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND> v584 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND>(v581, v582, v583);
      TLOAD(v579, v584);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
      // pto: %t__ci_tmp_v1
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v585 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v89);
      // pto: %t__ci_tmp_v1
      ;
      uint64_t v586 = (uint64_t) v40;
      TASSIGN(v585, v586);
      // pto: %t__tmp_v443
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v587 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v74);
      // pto: %t__tmp_v443
      ;
      uint64_t v588 = (uint64_t) v56;
      TASSIGN(v587, v588);
      // pto: %148
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v589;
      TRESHAPE(v589, v585);
      // pto: %149
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v590;
      TRESHAPE(v590, v587);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v590, v77, v589);
      // pto: %gate_ids_inline1667__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v591 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v74);
      // pto: %gate_ids_inline1667__ssa_v0
      ;
      uint64_t v592 = (uint64_t) v40;
      TASSIGN(v591, v592);
      pipe_barrier(PIPE_V);
      RoundMode v593 = RoundMode::CAST_ROUND;
      SaturationMode v594 = SaturationMode::ON;
      TCVT(v591, v587, v593, v594);
      // pto: %t__tmp_v444
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v595 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v74);
      // pto: %t__tmp_v444
      ;
      uint64_t v596 = (uint64_t) v40;
      TASSIGN(v595, v596);
      pipe_barrier(PIPE_V);
      TMULS(v595, v591, v95);
      // pto: %gate_scale_ids_inline1920__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v597 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v74);
      // pto: %gate_scale_ids_inline1920__ssa_v0
      ;
      uint64_t v598 = (uint64_t) v40;
      TASSIGN(v597, v598);
      pipe_barrier(PIPE_V);
      RoundMode v599 = RoundMode::CAST_FLOOR;
      SaturationMode v600 = SaturationMode::ON;
      TCVT(v597, v595, v599, v600);
      // pto: %gate_tmp_inline1922__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v601 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v74);
      // pto: %gate_tmp_inline1922__ssa_v0
      ;
      uint64_t v602 = (uint64_t) v43;
      TASSIGN(v601, v602);
      // pto: %gate_scale_vec_inline1914__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v603 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v74);
      // pto: %gate_scale_vec_inline1914__ssa_v0
      ;
      uint64_t v604 = (uint64_t) v57;
      TASSIGN(v603, v604);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
      TGATHER(v603, v579, v597, v601);
      // pto: %t__tmp_v445
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v605 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v74);
      // pto: %t__tmp_v445
      ;
      uint64_t v606 = (uint64_t) v37;
      TASSIGN(v605, v606);
      // pto: %150
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v607;
      // pto: %150
      ;
      uint64_t v608 = (uint64_t) v37;
      TASSIGN(v607, v608);
      // pto: %gate_raw_inline1658__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v609 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v74);
      // pto: %gate_raw_inline1658__ssa_v0
      ;
      uint64_t v610 = (uint64_t) v37;
      TASSIGN(v609, v610);
      TADDS(v609, v607, v80);
      // pto: %gate_norm_inline1766__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v611 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v74);
      // pto: %gate_norm_inline1766__ssa_v0
      ;
      uint64_t v612 = (uint64_t) v37;
      TASSIGN(v611, v612);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v611, v609, v571);
      // pto: %gate_scaled_inline1927__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v613 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v74);
      // pto: %gate_scaled_inline1927__ssa_v0
      ;
      uint64_t v614 = (uint64_t) v37;
      TASSIGN(v613, v614);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v613, v611, v603);
      // pto: %t__tmp_v446
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v615 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v74);
      // pto: %t__tmp_v446
      ;
      uint64_t v616 = (uint64_t) v58;
      TASSIGN(v615, v616);
      TCOLEXPAND(v615, v573);
      // pto: %gate_logits_inline1931__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v617 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v74);
      // pto: %gate_logits_inline1931__ssa_v0
      ;
      uint64_t v618 = (uint64_t) v37;
      TASSIGN(v617, v618);
      pipe_barrier(PIPE_V);
      TADD(v617, v613, v615);
      // pto: %t__tmp_v447
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v619 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v74);
      // pto: %t__tmp_v447
      ;
      uint64_t v620 = (uint64_t) v37;
      TASSIGN(v619, v620);
      pipe_barrier(PIPE_V);
      TNEG(v619, v617);
      // pto: %t__tmp_v448
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v621 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v74);
      // pto: %t__tmp_v448
      ;
      uint64_t v622 = (uint64_t) v37;
      TASSIGN(v621, v622);
      pipe_barrier(PIPE_V);
      TEXP(v621, v619);
      // pto: %t__tmp_v449
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v623 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v74);
      // pto: %t__tmp_v449
      ;
      uint64_t v624 = (uint64_t) v37;
      TASSIGN(v623, v624);
      pipe_barrier(PIPE_V);
      TADDS(v623, v621, v96);
      // pto: %gate_sigmoid_inline1689__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v625 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74, v74);
      // pto: %gate_sigmoid_inline1689__ssa_v0
      ;
      uint64_t v626 = (uint64_t) v59;
      TASSIGN(v625, v626);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
      TRECIP(v625, v623);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      // pto: %t__ci_tmp_v2
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v627 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v89);
      // pto: %t__ci_tmp_v2
      ;
      uint64_t v628 = (uint64_t) v37;
      TASSIGN(v627, v628);
      // pto: %t__tmp_v450
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v629 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v69);
      // pto: %t__tmp_v450
      ;
      uint64_t v630 = (uint64_t) v46;
      TASSIGN(v629, v630);
      // pto: %151
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v631;
      TRESHAPE(v631, v627);
      // pto: %152
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v632;
      TRESHAPE(v632, v629);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v632, v77, v631);
      // pto: %post_ids_float_inline1647__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v633 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v69);
      // pto: %post_ids_float_inline1647__ssa_v0
      ;
      uint64_t v634 = (uint64_t) v37;
      TASSIGN(v633, v634);
      pipe_barrier(PIPE_V);
      RoundMode v635 = RoundMode::CAST_ROUND;
      SaturationMode v636 = SaturationMode::ON;
      TCVT(v633, v629, v635, v636);
      // pto: %t__tmp_v451
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v637 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v69);
      // pto: %t__tmp_v451
      ;
      uint64_t v638 = (uint64_t) v60;
      TASSIGN(v637, v638);
      pipe_barrier(PIPE_V);
      TMULS(v637, v633, v95);
      // pto: %t__tmp_v452
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v639 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v69);
      // pto: %t__tmp_v452
      ;
      uint64_t v640 = (uint64_t) v60;
      TASSIGN(v639, v640);
      pipe_barrier(PIPE_V);
      RoundMode v641 = RoundMode::CAST_FLOOR;
      SaturationMode v642 = SaturationMode::ON;
      TCVT(v639, v637, v641, v642);
      // pto: %post_rows_inline1646__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v643 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v69);
      // pto: %post_rows_inline1646__ssa_v0
      ;
      uint64_t v644 = (uint64_t) v60;
      TASSIGN(v643, v644);
      pipe_barrier(PIPE_V);
      RoundMode v645 = RoundMode::CAST_ROUND;
      SaturationMode v646 = SaturationMode::ON;
      TCVT(v643, v639, v645, v646);
      // pto: %t__tmp_v453
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v647 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v69);
      // pto: %t__tmp_v453
      ;
      uint64_t v648 = (uint64_t) v60;
      TASSIGN(v647, v648);
      pipe_barrier(PIPE_V);
      TMULS(v647, v643, v97);
      // pto: %t__tmp_v454
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v649 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v69);
      // pto: %t__tmp_v454
      ;
      uint64_t v650 = (uint64_t) v37;
      TASSIGN(v649, v650);
      pipe_barrier(PIPE_V);
      TADD(v649, v633, v647);
      // pto: %t__tmp_v455
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v651 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v69);
      // pto: %t__tmp_v455
      ;
      uint64_t v652 = (uint64_t) v37;
      TASSIGN(v651, v652);
      pipe_barrier(PIPE_V);
      TADDS(v651, v649, v97);
      // pto: %post_ids_inline1645__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v653 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v69);
      // pto: %post_ids_inline1645__ssa_v0
      ;
      uint64_t v654 = (uint64_t) v37;
      TASSIGN(v653, v654);
      pipe_barrier(PIPE_V);
      RoundMode v655 = RoundMode::CAST_ROUND;
      SaturationMode v656 = SaturationMode::ON;
      TCVT(v653, v651, v655, v656);
      // pto: %post_tmp_inline1644__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v657 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v69);
      // pto: %post_tmp_inline1644__ssa_v0
      ;
      uint64_t v658 = (uint64_t) v60;
      TASSIGN(v657, v658);
      // pto: %t__tmp_v456
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v659 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v78);
      // pto: %t__tmp_v456
      ;
      uint64_t v660 = (uint64_t) v59;
      TASSIGN(v659, v660);
      // pto: %post_values_inline1715__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v661 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v69);
      // pto: %post_values_inline1715__ssa_v0
      ;
      uint64_t v662 = (uint64_t) v58;
      TASSIGN(v661, v662);
      pipe_barrier(PIPE_V);
      TGATHER(v661, v659, v653, v657);
      // pto: %t__tmp_v457
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v663 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v69);
      // pto: %t__tmp_v457
      ;
      uint64_t v664 = (uint64_t) v37;
      TASSIGN(v663, v664);
      pipe_barrier(PIPE_V);
      TMULS(v663, v661, v98);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      // pto: %153
      ;
      int64_t v665 = (int64_t) ((uint64_t) v511 * (uint64_t) v82);
      v663.SetValidShape(v70, v665);
      // pto: %154
      ;
      bool v666 = v489 == v81;
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
      if (v666) {
        // pto: %156
        ;
        int64_t v667 = (int64_t) ((uint64_t) v509 * (uint64_t) v82);
        // pto: %157
        ;
        int64_t v668 = v667 < v81 ? v81 : v667;
        // pto: %post_flat_inline1702__ssa_v0_pview
        ;
        const int64_t v669 = 0;
        // pto: %post_flat_inline1702__ssa_v0_pview
        ;
        __gm__ float* v670 = PTOAS__GLOBAL_TENSOR_DATA(v168);
        // pto: %post_flat_inline1702__ssa_v0_pview
        ;
        const int64_t v671 = 1;
        // pto: %post_flat_inline1702__ssa_v0_pview
        ;
        const int64_t v672 = 1;
        // pto: %post_flat_inline1702__ssa_v0_pview
        ;
        const int64_t v673 = 1;
        // pto: %post_flat_inline1702__ssa_v0_pview
        ;
        int64_t v674 = v70 * v160;
        // pto: %post_flat_inline1702__ssa_v0_pview
        ;
        int64_t v675 = v673 * v674;
        // pto: %post_flat_inline1702__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, -1> v676 = pto::Shape<1, 1, 1, 1, -1>(v671, v672, v673, v70, v665);
        // pto: %post_flat_inline1702__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v677 = pto::Stride<-1, -1, -1, -1, -1>(v672 * v675, v675, v674, v160, v70);
        // pto: %post_flat_inline1702__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v678 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v670 + (v669 + v81 * v160 + v668 * v70), v676, v677);
        TSTORE(v678, v663);
      };
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      // pto: %158
      ;
      int64_t v679 = v511 / v83;
      // pto: %159
      ;
      int64_t v680 = (int64_t) ((uint64_t) v679 * (uint64_t) v83);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      for (int64_t v681 = v81; v681 < v680; v681 += v83) {
        // pto: %160
        ;
        int64_t v682 = (int64_t) ((uint64_t) v509 + (uint64_t) v681);
        // pto: %161
        ;
        int64_t v683 = (int64_t) ((uint64_t) v681 + (uint64_t) v70);
        // pto: %162
        ;
        int64_t v684 = (int64_t) ((uint64_t) v509 + (uint64_t) v683);
        // pto: %flat_offset_mul
        ;
        int64_t v685 = (int64_t) ((uint64_t) v681 * (uint64_t) v74);
        // pto: %t__tmp_v458
        ;
        float v686 = v625.GetValue(v685);
        // pto: %163
        ;
        float v687 = v686 + v87;
        // pto: %165, %t__tmp_v459
        ;
        float v688 = v625.GetValue((int64_t) ((uint64_t) v685 + (uint64_t) v70));
        // pto: %166
        ;
        float v689 = v688 + v87;
        // pto: %168, %t__tmp_v460
        ;
        float v690 = v625.GetValue((int64_t) ((uint64_t) v685 + (uint64_t) v83));
        // pto: %169
        ;
        float v691 = v690 + v87;
        // pto: %171, %t__tmp_v461
        ;
        float v692 = v625.GetValue((int64_t) ((uint64_t) v685 + (uint64_t) v73));
        // pto: %172
        ;
        float v693 = v692 + v87;
        // pto: %t__tmp_v462
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v694 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %t__tmp_v462
        ;
        uint64_t v695 = (uint64_t) v61;
        TASSIGN(v694, v695);
        // pto: %173
        ;
        int64_t v696 = v682 < v81 ? v81 : v682;
        // pto: %174
        ;
        __gm__ bfloat16_t* v697 = PTOAS__GLOBAL_TENSOR_DATA(v116);
        // pto: %174
        ;
        const int64_t v698 = 0;
        // pto: %174
        ;
        const int64_t v699 = 16384;
        // pto: %174
        ;
        pto::Shape<1, 1, 1, 1, 4096> v700 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %174
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v701 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %174
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v702 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v697 + (v698 + v696 * v699), v700, v701);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v694, v702);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %t__tmp_v463
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v703 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %t__tmp_v463
        ;
        uint64_t v704 = (uint64_t) v62;
        TASSIGN(v703, v704);
        // pto: %176
        ;
        __gm__ bfloat16_t* v705 = PTOAS__GLOBAL_TENSOR_DATA(v116);
        // pto: %176
        ;
        const int64_t v706 = 4096;
        // pto: %176
        ;
        const int64_t v707 = 16384;
        // pto: %176
        ;
        pto::Shape<1, 1, 1, 1, 4096> v708 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %176
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v709 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %176
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v710 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v705 + (v706 + v696 * v707), v708, v709);
        TLOAD(v703, v710);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %t__tmp_v464
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v711 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %t__tmp_v464
        ;
        uint64_t v712 = (uint64_t) v63;
        TASSIGN(v711, v712);
        // pto: %178
        ;
        __gm__ bfloat16_t* v713 = PTOAS__GLOBAL_TENSOR_DATA(v116);
        // pto: %178
        ;
        const int64_t v714 = 8192;
        // pto: %178
        ;
        const int64_t v715 = 16384;
        // pto: %178
        ;
        pto::Shape<1, 1, 1, 1, 4096> v716 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %178
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v717 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %178
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v718 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v713 + (v714 + v696 * v715), v716, v717);
        TLOAD(v711, v718);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %t__tmp_v465
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v719 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %t__tmp_v465
        ;
        uint64_t v720 = (uint64_t) v64;
        TASSIGN(v719, v720);
        // pto: %180
        ;
        __gm__ bfloat16_t* v721 = PTOAS__GLOBAL_TENSOR_DATA(v116);
        // pto: %180
        ;
        const int64_t v722 = 12288;
        // pto: %180
        ;
        const int64_t v723 = 16384;
        // pto: %180
        ;
        pto::Shape<1, 1, 1, 1, 4096> v724 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %180
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v725 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %180
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v726 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v721 + (v722 + v696 * v723), v724, v725);
        TLOAD(v719, v726);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %183
        ;
        int64_t v727 = (int64_t) ((uint64_t) v683 * (uint64_t) v74);
        // pto: %181
        ;
        float v728 = v625.GetValue(v727);
        // pto: %185
        ;
        float v729 = v728 + v87;
        // pto: %189, %186
        ;
        float v730 = v625.GetValue((int64_t) ((uint64_t) v727 + (uint64_t) v70));
        // pto: %190
        ;
        float v731 = v730 + v87;
        // pto: %194, %191
        ;
        float v732 = v625.GetValue((int64_t) ((uint64_t) v727 + (uint64_t) v83));
        // pto: %195
        ;
        float v733 = v732 + v87;
        // pto: %199, %196
        ;
        float v734 = v625.GetValue((int64_t) ((uint64_t) v727 + (uint64_t) v73));
        // pto: %200
        ;
        float v735 = v734 + v87;
        // pto: %14
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v736 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %14
        ;
        uint64_t v737 = (uint64_t) v35;
        TASSIGN(v736, v737);
        // pto: %201
        ;
        int64_t v738 = v684 < v81 ? v81 : v684;
        // pto: %202
        ;
        __gm__ bfloat16_t* v739 = PTOAS__GLOBAL_TENSOR_DATA(v116);
        // pto: %202
        ;
        const int64_t v740 = 0;
        // pto: %202
        ;
        const int64_t v741 = 16384;
        // pto: %202
        ;
        pto::Shape<1, 1, 1, 1, 4096> v742 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %202
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v743 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %202
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v744 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v739 + (v740 + v738 * v741), v742, v743);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        TLOAD(v736, v744);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        // pto: %15
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v745 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %15
        ;
        uint64_t v746 = (uint64_t) v65;
        TASSIGN(v745, v746);
        // pto: %204
        ;
        __gm__ bfloat16_t* v747 = PTOAS__GLOBAL_TENSOR_DATA(v116);
        // pto: %204
        ;
        const int64_t v748 = 4096;
        // pto: %204
        ;
        const int64_t v749 = 16384;
        // pto: %204
        ;
        pto::Shape<1, 1, 1, 1, 4096> v750 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %204
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v751 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %204
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v752 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v747 + (v748 + v738 * v749), v750, v751);
        TLOAD(v745, v752);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        // pto: %16
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v753 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %16
        ;
        uint64_t v754 = (uint64_t) v66;
        TASSIGN(v753, v754);
        // pto: %206
        ;
        __gm__ bfloat16_t* v755 = PTOAS__GLOBAL_TENSOR_DATA(v116);
        // pto: %206
        ;
        const int64_t v756 = 8192;
        // pto: %206
        ;
        const int64_t v757 = 16384;
        // pto: %206
        ;
        pto::Shape<1, 1, 1, 1, 4096> v758 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %206
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v759 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %206
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v760 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v755 + (v756 + v738 * v757), v758, v759);
        TLOAD(v753, v760);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        // pto: %17
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v761 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %17
        ;
        uint64_t v762 = (uint64_t) v67;
        TASSIGN(v761, v762);
        // pto: %208
        ;
        __gm__ bfloat16_t* v763 = PTOAS__GLOBAL_TENSOR_DATA(v116);
        // pto: %208
        ;
        const int64_t v764 = 12288;
        // pto: %208
        ;
        const int64_t v765 = 16384;
        // pto: %208
        ;
        pto::Shape<1, 1, 1, 1, 4096> v766 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %208
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v767 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %208
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v768 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v763 + (v764 + v738 * v765), v766, v767);
        TLOAD(v761, v768);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        // pto: %scalar_x0_inline1625__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v769 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %scalar_x0_inline1625__ssa_v0
        ;
        uint64_t v770 = (uint64_t) v37;
        TASSIGN(v769, v770);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        RoundMode v771 = RoundMode::CAST_ROUND;
        SaturationMode v772 = SaturationMode::ON;
        TCVT(v769, v694, v771, v772);
        // pto: %scalar_y0_inline1769__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v773 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %scalar_y0_inline1769__ssa_v0
        ;
        uint64_t v774 = (uint64_t) v37;
        TASSIGN(v773, v774);
        pipe_barrier(PIPE_V);
        TMULS(v773, v769, v687);
        // pto: %scalar_x1_inline1663__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v775 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %scalar_x1_inline1663__ssa_v0
        ;
        uint64_t v776 = (uint64_t) v61;
        TASSIGN(v775, v776);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        RoundMode v777 = RoundMode::CAST_ROUND;
        SaturationMode v778 = SaturationMode::ON;
        TCVT(v775, v703, v777, v778);
        // pto: %scalar_y1_inline1655__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v779 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %scalar_y1_inline1655__ssa_v0
        ;
        uint64_t v780 = (uint64_t) v61;
        TASSIGN(v779, v780);
        pipe_barrier(PIPE_V);
        TMULS(v779, v775, v689);
        // pto: %scalar_sum01_inline1872__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v781 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %scalar_sum01_inline1872__ssa_v0
        ;
        uint64_t v782 = (uint64_t) v37;
        TASSIGN(v781, v782);
        pipe_barrier(PIPE_V);
        TADD(v781, v773, v779);
        // pto: %scalar_x2_inline1624__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v783 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %scalar_x2_inline1624__ssa_v0
        ;
        uint64_t v784 = (uint64_t) v61;
        TASSIGN(v783, v784);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        RoundMode v785 = RoundMode::CAST_ROUND;
        SaturationMode v786 = SaturationMode::ON;
        TCVT(v783, v711, v785, v786);
        // pto: %scalar_y2_inline1669__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v787 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %scalar_y2_inline1669__ssa_v0
        ;
        uint64_t v788 = (uint64_t) v61;
        TASSIGN(v787, v788);
        pipe_barrier(PIPE_V);
        TMULS(v787, v783, v691);
        // pto: %scalar_x3_inline1708__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v789 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %scalar_x3_inline1708__ssa_v0
        ;
        uint64_t v790 = (uint64_t) v62;
        TASSIGN(v789, v790);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        RoundMode v791 = RoundMode::CAST_ROUND;
        SaturationMode v792 = SaturationMode::ON;
        TCVT(v789, v719, v791, v792);
        // pto: %scalar_y3_inline1623__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v793 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %scalar_y3_inline1623__ssa_v0
        ;
        uint64_t v794 = (uint64_t) v62;
        TASSIGN(v793, v794);
        pipe_barrier(PIPE_V);
        TMULS(v793, v789, v693);
        // pto: %t__tmp_v466
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v795 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %t__tmp_v466
        ;
        uint64_t v796 = (uint64_t) v61;
        TASSIGN(v795, v796);
        pipe_barrier(PIPE_V);
        TADD(v795, v787, v793);
        // pto: %scalar_result_inline1742__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v797 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %scalar_result_inline1742__ssa_v0
        ;
        uint64_t v798 = (uint64_t) v37;
        TASSIGN(v797, v798);
        pipe_barrier(PIPE_V);
        TADD(v797, v781, v795);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %scalar_bf16_inline1814__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v799 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %scalar_bf16_inline1814__ssa_v0
        ;
        uint64_t v800 = (uint64_t) v37;
        TASSIGN(v799, v800);
        pipe_barrier(PIPE_V);
        RoundMode v801 = RoundMode::CAST_RINT;
        SaturationMode v802 = SaturationMode::ON;
        TCVT(v799, v797, v801, v802);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        // pto: %mixed_view_inline1804__ssa_v0_pview
        ;
        __gm__ bfloat16_t* v803 = PTOAS__GLOBAL_TENSOR_DATA(v177);
        // pto: %mixed_view_inline1804__ssa_v0_pview
        ;
        const int64_t v804 = 0;
        // pto: %mixed_view_inline1804__ssa_v0_pview
        ;
        const int64_t v805 = 4096;
        // pto: %mixed_view_inline1804__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, 4096> v806 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %mixed_view_inline1804__ssa_v0_pview
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v807 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %mixed_view_inline1804__ssa_v0_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v808 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v803 + (v804 + v696 * v805), v806, v807);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v808, v799);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        // pto: %18
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v809 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %18
        ;
        uint64_t v810 = (uint64_t) v68;
        TASSIGN(v809, v810);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
        RoundMode v811 = RoundMode::CAST_ROUND;
        SaturationMode v812 = SaturationMode::ON;
        TCVT(v809, v736, v811, v812);
        // pto: %19
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v813 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %19
        ;
        uint64_t v814 = (uint64_t) v68;
        TASSIGN(v813, v814);
        pipe_barrier(PIPE_V);
        TMULS(v813, v809, v729);
        // pto: %20
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v815 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %20
        ;
        uint64_t v816 = (uint64_t) v35;
        TASSIGN(v815, v816);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        RoundMode v817 = RoundMode::CAST_ROUND;
        SaturationMode v818 = SaturationMode::ON;
        TCVT(v815, v745, v817, v818);
        // pto: %21
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v819 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %21
        ;
        uint64_t v820 = (uint64_t) v35;
        TASSIGN(v819, v820);
        pipe_barrier(PIPE_V);
        TMULS(v819, v815, v731);
        // pto: %22
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v821 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %22
        ;
        uint64_t v822 = (uint64_t) v68;
        TASSIGN(v821, v822);
        pipe_barrier(PIPE_V);
        TADD(v821, v813, v819);
        // pto: %23
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v823 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %23
        ;
        uint64_t v824 = (uint64_t) v35;
        TASSIGN(v823, v824);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        RoundMode v825 = RoundMode::CAST_ROUND;
        SaturationMode v826 = SaturationMode::ON;
        TCVT(v823, v753, v825, v826);
        // pto: %24
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v827 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %24
        ;
        uint64_t v828 = (uint64_t) v35;
        TASSIGN(v827, v828);
        pipe_barrier(PIPE_V);
        TMULS(v827, v823, v733);
        // pto: %25
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v829 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %25
        ;
        uint64_t v830 = (uint64_t) v65;
        TASSIGN(v829, v830);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        RoundMode v831 = RoundMode::CAST_ROUND;
        SaturationMode v832 = SaturationMode::ON;
        TCVT(v829, v761, v831, v832);
        // pto: %26
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v833 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %26
        ;
        uint64_t v834 = (uint64_t) v65;
        TASSIGN(v833, v834);
        pipe_barrier(PIPE_V);
        TMULS(v833, v829, v735);
        // pto: %27
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v835 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %27
        ;
        uint64_t v836 = (uint64_t) v35;
        TASSIGN(v835, v836);
        pipe_barrier(PIPE_V);
        TADD(v835, v827, v833);
        // pto: %28
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v837 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %28
        ;
        uint64_t v838 = (uint64_t) v68;
        TASSIGN(v837, v838);
        pipe_barrier(PIPE_V);
        TADD(v837, v821, v835);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        // pto: %29
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v839 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %29
        ;
        uint64_t v840 = (uint64_t) v68;
        TASSIGN(v839, v840);
        pipe_barrier(PIPE_V);
        RoundMode v841 = RoundMode::CAST_RINT;
        SaturationMode v842 = SaturationMode::ON;
        TCVT(v839, v837, v841, v842);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %212
        ;
        __gm__ bfloat16_t* v843 = PTOAS__GLOBAL_TENSOR_DATA(v177);
        // pto: %212
        ;
        const int64_t v844 = 0;
        // pto: %212
        ;
        const int64_t v845 = 4096;
        // pto: %212
        ;
        pto::Shape<1, 1, 1, 1, 4096> v846 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %212
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v847 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %212
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v848 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v843 + (v844 + v738 * v845), v846, v847);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v848, v839);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
      };
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_S, PIPE_V, EVENT_ID1);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      // pto: %213, %214
      ;
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      if ((int64_t) ((uint64_t) v511 - (uint64_t) v680) == v70) {
        // pto: %215
        ;
        int64_t v849 = (int64_t) ((uint64_t) v509 + (uint64_t) v680);
        int64_t v850 = (int64_t) ((uint64_t) v679 * (uint64_t) v75);
        // pto: %216
        ;
        float v851 = v625.GetValue(v850);
        // pto: %219
        ;
        float v852 = v851 + v87;
        // pto: %222, %220
        ;
        float v853 = v625.GetValue((int64_t) ((uint64_t) v850 + (uint64_t) v70));
        // pto: %223
        ;
        float v854 = v853 + v87;
        // pto: %226, %224
        ;
        float v855 = v625.GetValue((int64_t) ((uint64_t) v850 + (uint64_t) v83));
        // pto: %227
        ;
        float v856 = v855 + v87;
        // pto: %230, %228
        ;
        float v857 = v625.GetValue((int64_t) ((uint64_t) v850 + (uint64_t) v73));
        // pto: %231
        ;
        float v858 = v857 + v87;
        // pto: %30
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v859 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %30
        ;
        uint64_t v860 = (uint64_t) v61;
        TASSIGN(v859, v860);
        // pto: %232
        ;
        int64_t v861 = v849 < v81 ? v81 : v849;
        // pto: %233
        ;
        __gm__ bfloat16_t* v862 = PTOAS__GLOBAL_TENSOR_DATA(v116);
        // pto: %233
        ;
        const int64_t v863 = 0;
        // pto: %233
        ;
        const int64_t v864 = 16384;
        // pto: %233
        ;
        pto::Shape<1, 1, 1, 1, 4096> v865 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %233
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v866 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %233
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v867 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v862 + (v863 + v861 * v864), v865, v866);
        TLOAD(v859, v867);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %31
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v868 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %31
        ;
        uint64_t v869 = (uint64_t) v37;
        TASSIGN(v868, v869);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v870 = RoundMode::CAST_ROUND;
        SaturationMode v871 = SaturationMode::ON;
        TCVT(v868, v859, v870, v871);
        // pto: %32
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v872 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %32
        ;
        uint64_t v873 = (uint64_t) v37;
        TASSIGN(v872, v873);
        pipe_barrier(PIPE_V);
        TMULS(v872, v868, v852);
        // pto: %33
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v874 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %33
        ;
        uint64_t v875 = (uint64_t) v62;
        TASSIGN(v874, v875);
        // pto: %235
        ;
        __gm__ bfloat16_t* v876 = PTOAS__GLOBAL_TENSOR_DATA(v116);
        // pto: %235
        ;
        const int64_t v877 = 4096;
        // pto: %235
        ;
        const int64_t v878 = 16384;
        // pto: %235
        ;
        pto::Shape<1, 1, 1, 1, 4096> v879 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %235
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v880 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %235
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v881 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v876 + (v877 + v861 * v878), v879, v880);
        TLOAD(v874, v881);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %34
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v882 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %34
        ;
        uint64_t v883 = (uint64_t) v61;
        TASSIGN(v882, v883);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v884 = RoundMode::CAST_ROUND;
        SaturationMode v885 = SaturationMode::ON;
        TCVT(v882, v874, v884, v885);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %35
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v886 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %35
        ;
        uint64_t v887 = (uint64_t) v61;
        TASSIGN(v886, v887);
        pipe_barrier(PIPE_V);
        TMULS(v886, v882, v854);
        // pto: %36
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v888 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %36
        ;
        uint64_t v889 = (uint64_t) v37;
        TASSIGN(v888, v889);
        pipe_barrier(PIPE_V);
        TADD(v888, v872, v886);
        // pto: %37
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v890 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %37
        ;
        uint64_t v891 = (uint64_t) v62;
        TASSIGN(v890, v891);
        // pto: %237
        ;
        __gm__ bfloat16_t* v892 = PTOAS__GLOBAL_TENSOR_DATA(v116);
        // pto: %237
        ;
        const int64_t v893 = 8192;
        // pto: %237
        ;
        const int64_t v894 = 16384;
        // pto: %237
        ;
        pto::Shape<1, 1, 1, 1, 4096> v895 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %237
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v896 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %237
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v897 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v892 + (v893 + v861 * v894), v895, v896);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v890, v897);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %38
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v898 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %38
        ;
        uint64_t v899 = (uint64_t) v61;
        TASSIGN(v898, v899);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v900 = RoundMode::CAST_ROUND;
        SaturationMode v901 = SaturationMode::ON;
        TCVT(v898, v890, v900, v901);
        // pto: %39
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v902 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %39
        ;
        uint64_t v903 = (uint64_t) v61;
        TASSIGN(v902, v903);
        pipe_barrier(PIPE_V);
        TMULS(v902, v898, v856);
        // pto: %40
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v904 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %40
        ;
        uint64_t v905 = (uint64_t) v68;
        TASSIGN(v904, v905);
        // pto: %239
        ;
        __gm__ bfloat16_t* v906 = PTOAS__GLOBAL_TENSOR_DATA(v116);
        // pto: %239
        ;
        const int64_t v907 = 12288;
        // pto: %239
        ;
        const int64_t v908 = 16384;
        // pto: %239
        ;
        pto::Shape<1, 1, 1, 1, 4096> v909 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %239
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v910 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %239
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v911 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v906 + (v907 + v861 * v908), v909, v910);
        TLOAD(v904, v911);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %41
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v912 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %41
        ;
        uint64_t v913 = (uint64_t) v62;
        TASSIGN(v912, v913);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v914 = RoundMode::CAST_ROUND;
        SaturationMode v915 = SaturationMode::ON;
        TCVT(v912, v904, v914, v915);
        // pto: %42
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v916 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %42
        ;
        uint64_t v917 = (uint64_t) v62;
        TASSIGN(v916, v917);
        pipe_barrier(PIPE_V);
        TMULS(v916, v912, v858);
        // pto: %43
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v918 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %43
        ;
        uint64_t v919 = (uint64_t) v61;
        TASSIGN(v918, v919);
        pipe_barrier(PIPE_V);
        TADD(v918, v902, v916);
        // pto: %44
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v920 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %44
        ;
        uint64_t v921 = (uint64_t) v37;
        TASSIGN(v920, v921);
        pipe_barrier(PIPE_V);
        TADD(v920, v888, v918);
        // pto: %45
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v922 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
        // pto: %45
        ;
        uint64_t v923 = (uint64_t) v37;
        TASSIGN(v922, v923);
        pipe_barrier(PIPE_V);
        RoundMode v924 = RoundMode::CAST_RINT;
        SaturationMode v925 = SaturationMode::ON;
        TCVT(v922, v920, v924, v925);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %242
        ;
        __gm__ bfloat16_t* v926 = PTOAS__GLOBAL_TENSOR_DATA(v177);
        // pto: %242
        ;
        const int64_t v927 = 0;
        // pto: %242
        ;
        const int64_t v928 = 4096;
        // pto: %242
        ;
        pto::Shape<1, 1, 1, 1, 4096> v929 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %242
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v930 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %242
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v931 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v926 + (v927 + v861 * v928), v929, v930);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v931, v922);
      };
      set_flag(PIPE_S, PIPE_V, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    };
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  }
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}