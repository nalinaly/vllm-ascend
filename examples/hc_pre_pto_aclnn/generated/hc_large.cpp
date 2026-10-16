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
  // pto: %mixes_partials_inline1690__ssa_v0_view
  const int64_t v56 = 1;
  // pto: %mixes_partials_inline1690__ssa_v0_view
  const int64_t v57 = 1;
  // pto: %mixes_partials_inline1690__ssa_v0_view
  const int64_t v58 = 1;
  // pto: %mixes_partials_inline1690__ssa_v0_view
  int64_t v59 = v55 * v41;
  // pto: %mixes_partials_inline1690__ssa_v0_view
  int64_t v60 = v58 * v59;
  // pto: %mixes_partials_inline1690__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v61 = pto::Shape<1, 1, 1, -1, -1>(v56, v57, v58, v55, v41);
  // pto: %mixes_partials_inline1690__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v62 = pto::Stride<-1, -1, -1, -1, -1>(v57 * v60, v60, v59, v41, v42);
  // pto: %mixes_partials_inline1690__ssa_v0_view
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
  // pto: %mid_stream_product_inline1735__phi_v5
  Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v72 = Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v46, v43);
  // pto: %mid_stream_product_inline1735__phi_v5
  uint64_t v73 = (uint64_t) v35;
  TASSIGN(v72, v73);
  const int32_t v74 = 0;
  auto v75 = TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>(v13, v74, v34);
  // pto: %worker_inline1772__ssa_v0
  // pto: %linear_workers_inline1699__ssa_v0_idx
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
    // pto: %mid_stream_product_inline1735__tile
    ;
    Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v80 = Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v46, v41);
    // pto: %mid_stream_product_inline1735__tile
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
      // pto: %mid_weight_chunk_inline1763__tile
      ;
      Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v86 = Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v43, v51);
      // pto: %mid_weight_chunk_inline1763__tile
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
      // pto: %mid_gathered_chunk_inline1673__tile
      ;
      const int32_t v112 = 64;
      // pto: %mid_gathered_chunk_inline1673__tile
      ;
      const int32_t v113 = 512;
      // pto: %mid_gathered_chunk_inline1673__tile
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
        // pto: %mid_weight_chunk_inline1763__tile_t
        ;
        Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v115 = Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v51, v43);
        // pto: %mid_weight_chunk_inline1763__tile_t
        ;
        uint64_t v116 = (uint64_t) v36;
        TASSIGN(v115, v116);
        // pto: %mid_stream_product_inline1735__tile_l0_init_storage
        ;
        Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v117 = Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v46, v41);
        // pto: %mid_stream_product_inline1735__tile_l0_init_storage
        ;
        uint64_t v118 = (uint64_t) v35;
        TASSIGN(v117, v118);
        v117.SetValidShape(v46, v43);
        for (int64_t v119 = v48; v119 < v51; v119 += v52) {
          // pto: %mid_stream_product_inline1735__tile_l0_a
          ;
          Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v120 = Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v46, v53);
          // pto: %mid_stream_product_inline1735__tile_l0_a
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
          // pto: %mid_stream_product_inline1735__tile_l0_b
          ;
          Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v126 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v125, v43);
          // pto: %mid_stream_product_inline1735__tile_l0_b
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
    // pto: %mixes_partials_inline1690__iter_v1_pview
    ;
    __gm__ float* v196 = PTOAS__GLOBAL_TENSOR_DATA(v63);
    // pto: %mixes_partials_inline1690__iter_v1_pview
    ;
    const int64_t v197 = 0;
    // pto: %mixes_partials_inline1690__iter_v1_pview
    ;
    const int64_t v198 = 32;
    // pto: %mixes_partials_inline1690__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 64, 32> v199 = pto::Shape<1, 1, 1, 64, 32>();
    // pto: %mixes_partials_inline1690__iter_v1_pview
    ;
    pto::Stride<2048, 2048, 2048, 32, 1> v200 = pto::Stride<2048, 2048, 2048, 32, 1>();
    // pto: %mixes_partials_inline1690__iter_v1_pview
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
  // pto: %c33920_i64
  const int64_t v39 = 33920;
  // pto: %c34176_i64
  const int64_t v40 = 34176;
  // pto: %c33952_i64
  const int64_t v41 = 33952;
  const int64_t v42 = 32928;
  // pto: %c33408_i64
  const int64_t v43 = 33408;
  // pto: %c33664_i64
  const int64_t v44 = 33664;
  // pto: %c34688_i64
  const int64_t v45 = 34688;
  // pto: %c35200_i64
  const int64_t v46 = 35200;
  // pto: %c35712_i64
  const int64_t v47 = 35712;
  // pto: %c36224_i64
  const int64_t v48 = 36224;
  // pto: %c36736_i64
  const int64_t v49 = 36736;
  // pto: %c37248_i64
  const int64_t v50 = 37248;
  // pto: %c34432_i64
  const int64_t v51 = 34432;
  // pto: %c34944_i64
  const int64_t v52 = 34944;
  // pto: %c34720_i64
  const int64_t v53 = 34720;
  // pto: %c34752_i64
  const int64_t v54 = 34752;
  // pto: %c34784_i64
  const int64_t v55 = 34784;
  // pto: %c33984_i64
  const int64_t v56 = 33984;
  // pto: %c33152_i64
  const int64_t v57 = 33152;
  // pto: %c147584_i64
  const int64_t v58 = 147584;
  // pto: %c33024_i64
  const int64_t v59 = 33024;
  // pto: %c49280_i64
  const int64_t v60 = 49280;
  // pto: %c65664_i64
  const int64_t v61 = 65664;
  // pto: %c73856_i64
  const int64_t v62 = 73856;
  // pto: %c131200_i64
  const int64_t v63 = 131200;
  // pto: %c114816_i64
  const int64_t v64 = 114816;
  // pto: %c123008_i64
  const int64_t v65 = 123008;
  // pto: %c139392_i64
  const int64_t v66 = 139392;
  // pto: %c82048_i64
  const int64_t v67 = 82048;
  // pto: %c32_index
  const int64_t v68 = 32;
  // pto: %c1_index
  const int64_t v69 = 1;
  // pto: %c24_index
  const int64_t v70 = 24;
  // pto: %c16384_index
  const int64_t v71 = 16384;
  // pto: %c3_index
  const int64_t v72 = 3;
  // pto: %c8_index
  const int64_t v73 = 8;
  const int64_t v74 = 16;
  // pto: %c4096_index
  const int64_t v75 = 4096;
  // pto: %c0_i32
  const int32_t v76 = 0;
  // pto: %c64_index
  const int64_t v77 = 64;
  // pto: %c2048_index
  const int64_t v78 = 2048;
  // pto: %cst_45
  const float v79 = 0.0f;
  // pto: %c0_index
  const int64_t v80 = 0;
  // pto: %c4_index
  const int64_t v81 = 4;
  // pto: %c2_index
  const int64_t v82 = 2;
  // pto: %c512_index
  const int64_t v83 = 512;
  // pto: %c256_index
  const int64_t v84 = 256;
  // pto: %cst_51
  const float v85 = 6.10351563E-5f;
  // pto: %cst_52
  const float v86 = 9.99999997E-7f;
  // pto: %c128_index
  const int64_t v87 = 128;
  // pto: %c192_index
  const int64_t v88 = 192;
  // pto: %c1_i32
  const int32_t v89 = 1;
  // pto: %c2_i32
  const int32_t v90 = 2;
  // pto: %c4_i32
  const int32_t v91 = 4;
  // pto: %c8_i32
  const int32_t v92 = 8;
  // pto: %c19_index
  const int64_t v93 = 19;
  // pto: %cst_60
  const float v94 = 0.25f;
  // pto: %cst_61
  const float v95 = 1.0f;
  // pto: %cst_62
  const float v96 = 4.0f;
  // pto: %cst_63
  const float v97 = 2.0f;
  // pto: %234
  int64_t v98 = (int64_t) ((uint64_t) v29 * (uint64_t) v73);
  // pto: %rms_partials_inline1751__ssa_v0_view
  const int64_t v99 = 1;
  // pto: %rms_partials_inline1751__ssa_v0_view
  const int64_t v100 = 1;
  // pto: %rms_partials_inline1751__ssa_v0_view
  const int64_t v101 = 1;
  // pto: %rms_partials_inline1751__ssa_v0_view
  int64_t v102 = v98 * v69;
  // pto: %rms_partials_inline1751__ssa_v0_view
  int64_t v103 = v101 * v102;
  // pto: %rms_partials_inline1751__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v104 = pto::Shape<1, 1, 1, -1, -1>(v99, v100, v101, v98, v69);
  // pto: %rms_partials_inline1751__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v105 = pto::Stride<-1, -1, -1, -1, -1>(v100 * v103, v103, v102, v69, v98);
  // pto: %rms_partials_inline1751__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v106 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v104, v105);
  // pto: %x_flat_inline1647__ssa_v0_view
  const int64_t v107 = 1;
  // pto: %x_flat_inline1647__ssa_v0_view
  const int64_t v108 = 1;
  // pto: %x_flat_inline1647__ssa_v0_view
  const int64_t v109 = 1;
  // pto: %x_flat_inline1647__ssa_v0_view
  int64_t v110 = (int64_t) v30;
  // pto: %x_flat_inline1647__ssa_v0_view
  int64_t v111 = v110 * v71;
  // pto: %x_flat_inline1647__ssa_v0_view
  int64_t v112 = v109 * v111;
  // pto: %x_flat_inline1647__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v113 = pto::Shape<1, 1, 1, -1, -1>(v107, v108, v109, v110, v71);
  // pto: %x_flat_inline1647__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v114 = pto::Stride<-1, -1, -1, -1, -1>(v108 * v112, v112, v111, v71, v69);
  // pto: %x_flat_inline1647__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v115 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v113, v114);
  // pto: %235
  int64_t v116 = (int64_t) ((uint64_t) v29 * (uint64_t) v68);
  // pto: %parts_2d_inline1805__ssa_v0_view
  const int64_t v117 = 1;
  // pto: %parts_2d_inline1805__ssa_v0_view
  const int64_t v118 = 1;
  // pto: %parts_2d_inline1805__ssa_v0_view
  const int64_t v119 = 1;
  // pto: %parts_2d_inline1805__ssa_v0_view
  int64_t v120 = v73 * v116;
  // pto: %parts_2d_inline1805__ssa_v0_view
  int64_t v121 = v119 * v120;
  // pto: %parts_2d_inline1805__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v122 = pto::Shape<1, 1, 1, -1, -1>(v117, v118, v119, v73, v116);
  // pto: %parts_2d_inline1805__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v123 = pto::Stride<-1, -1, -1, -1, -1>(v118 * v121, v121, v120, v116, v69);
  // pto: %parts_2d_inline1805__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v124 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v122, v123);
  // pto: %rms_2d_inline1755__ssa_v0_view
  const int64_t v125 = 1;
  // pto: %rms_2d_inline1755__ssa_v0_view
  const int64_t v126 = 1;
  // pto: %rms_2d_inline1755__ssa_v0_view
  const int64_t v127 = 1;
  // pto: %rms_2d_inline1755__ssa_v0_view
  int64_t v128 = (int64_t) v29;
  // pto: %rms_2d_inline1755__ssa_v0_view
  int64_t v129 = v73 * v128;
  // pto: %rms_2d_inline1755__ssa_v0_view
  int64_t v130 = v127 * v129;
  // pto: %rms_2d_inline1755__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v131 = pto::Shape<1, 1, 1, -1, -1>(v125, v126, v127, v73, (int64_t) v29);
  // pto: %rms_2d_inline1755__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v132 = pto::Stride<-1, -1, -1, -1, -1>(v126 * v130, v130, v129, v128, v69);
  // pto: %rms_2d_inline1755__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v133 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v131, v132);
  // pto: %hc_base_2d_inline1697__ssa_v0_view
  const int64_t v134 = 1;
  // pto: %hc_base_2d_inline1697__ssa_v0_view
  const int64_t v135 = 1;
  // pto: %hc_base_2d_inline1697__ssa_v0_view
  const int64_t v136 = 1;
  // pto: %hc_base_2d_inline1697__ssa_v0_view
  int64_t v137 = v69 * v70;
  // pto: %hc_base_2d_inline1697__ssa_v0_view
  int64_t v138 = v136 * v137;
  // pto: %hc_base_2d_inline1697__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v139 = pto::Shape<1, 1, 1, -1, -1>(v134, v135, v136, v69, v70);
  // pto: %hc_base_2d_inline1697__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v140 = pto::Stride<-1, -1, -1, -1, -1>(v135 * v138, v138, v137, v70, v69);
  // pto: %hc_base_2d_inline1697__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v141 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v8, v139, v140);
  // pto: %comb_inline1666__ssa_v0_view
  const int64_t v142 = 1;
  // pto: %comb_inline1666__ssa_v0_view
  const int64_t v143 = 1;
  // pto: %comb_inline1666__ssa_v0_view
  const int64_t v144 = 1;
  // pto: %comb_inline1666__ssa_v0_view
  int64_t v145 = (int64_t) v30;
  // pto: %comb_inline1666__ssa_v0_view
  int64_t v146 = v145 * v74;
  // pto: %comb_inline1666__ssa_v0_view
  int64_t v147 = v144 * v146;
  // pto: %comb_inline1666__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v148 = pto::Shape<1, 1, 1, -1, -1>(v142, v143, v144, v145, v74);
  // pto: %comb_inline1666__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v149 = pto::Stride<-1, -1, -1, -1, -1>(v143 * v147, v147, v146, v74, v69);
  // pto: %comb_inline1666__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v150 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v9, v148, v149);
  // pto: %hc_scale_2d_inline1720__ssa_v0_view
  const int64_t v151 = 1;
  // pto: %hc_scale_2d_inline1720__ssa_v0_view
  const int64_t v152 = 1;
  // pto: %hc_scale_2d_inline1720__ssa_v0_view
  const int64_t v153 = 1;
  // pto: %hc_scale_2d_inline1720__ssa_v0_view
  int64_t v154 = v69 * v72;
  // pto: %hc_scale_2d_inline1720__ssa_v0_view
  int64_t v155 = v153 * v154;
  // pto: %hc_scale_2d_inline1720__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v156 = pto::Shape<1, 1, 1, -1, -1>(v151, v152, v153, v69, v72);
  // pto: %hc_scale_2d_inline1720__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v157 = pto::Stride<-1, -1, -1, -1, -1>(v152 * v155, v155, v154, v72, v69);
  // pto: %hc_scale_2d_inline1720__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v158 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v10, v156, v157);
  // pto: %236
  int64_t v159 = (int64_t) ((uint64_t) v30 * (uint64_t) v81);
  // pto: %post_flat_inline1729__ssa_v0_view
  const int64_t v160 = 1;
  // pto: %post_flat_inline1729__ssa_v0_view
  const int64_t v161 = 1;
  // pto: %post_flat_inline1729__ssa_v0_view
  const int64_t v162 = 1;
  // pto: %post_flat_inline1729__ssa_v0_view
  int64_t v163 = v69 * v159;
  // pto: %post_flat_inline1729__ssa_v0_view
  int64_t v164 = v162 * v163;
  // pto: %post_flat_inline1729__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v165 = pto::Shape<1, 1, 1, -1, -1>(v160, v161, v162, v69, v159);
  // pto: %post_flat_inline1729__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v166 = pto::Stride<-1, -1, -1, -1, -1>(v161 * v164, v164, v163, v159, v69);
  // pto: %post_flat_inline1729__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v167 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v11, v165, v166);
  // pto: %mixed_view_inline1689__ssa_v0_view
  const int64_t v168 = 1;
  // pto: %mixed_view_inline1689__ssa_v0_view
  const int64_t v169 = 1;
  // pto: %mixed_view_inline1689__ssa_v0_view
  const int64_t v170 = 1;
  // pto: %mixed_view_inline1689__ssa_v0_view
  int64_t v171 = (int64_t) v30;
  // pto: %mixed_view_inline1689__ssa_v0_view
  int64_t v172 = v171 * v75;
  // pto: %mixed_view_inline1689__ssa_v0_view
  int64_t v173 = v170 * v172;
  // pto: %mixed_view_inline1689__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v174 = pto::Shape<1, 1, 1, -1, -1>(v168, v169, v170, v171, v75);
  // pto: %mixed_view_inline1689__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v175 = pto::Stride<-1, -1, -1, -1, -1>(v169 * v173, v173, v172, v75, v69);
  // pto: %mixed_view_inline1689__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v176 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v12, v174, v175);
  const int32_t v177 = 0;
  auto v178 = TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>(v13, v177, v76);
  // pto: %worker_inline1772__ssa_v0
  int64_t v179 = (int64_t) v31;
  // pto: %linear_workers_inline1699__ssa_v0_idx
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  set_flag(PIPE_S, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
  for (int64_t v180 = v179; v180 < v14; v180 += (int64_t) v15) {
    // pto: %46, %47
    ;
    int64_t v181 = (int64_t) ((uint64_t) (v180 / v73) * (uint64_t) v77);
    // pto: %48
    ;
    int64_t v182 = v180 % v73;
    // pto: %49
    ;
    int64_t v183 = (int64_t) ((uint64_t) v182 * (uint64_t) v78);
    // pto: %mid_stream_rms_total_inline1758__tile
    ;
    Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v184 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v68);
    // pto: %mid_stream_rms_total_inline1758__tile
    ;
    uint64_t v185 = (uint64_t) v34;
    TASSIGN(v184, v185);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TEXPANDS(v184, v79);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    for (int64_t v186 = v80; v186 < v81; v186 += v82) {
      // pto: %52
      ;
      int64_t v187 = (int64_t) ((uint64_t) v186 * (uint64_t) v83);
      // pto: %53
      ;
      int64_t v188 = (int64_t) ((uint64_t) v183 + (uint64_t) v187);
      // pto: %55, %lane_inline1664__ssa_v0, %54
      ;
      int64_t v189 = (int64_t) ((uint64_t) v181 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) v33) * (uint64_t) v68));
      // pto: %56
      ;
      int64_t v190 = (int64_t) ((uint64_t) v17 - (uint64_t) v189);
      // pto: %57
      ;
      int64_t v191 = v190 < v68 ? v190 : v68;
      // pto: %58
      ;
      int64_t v192 = v191 < v80 ? v80 : v191;
      // pto: %61, %60
      ;
      int64_t v193 = (int64_t) ((uint64_t) v183 + (uint64_t) ((int64_t) (uint64_t) v187 + (uint64_t) v83));
      // pto: %mid_stream_source_inline1660__tile
      ;
      Tile<TileType::Vec, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v194 = Tile<TileType::Vec, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v192, v83);
      // pto: %mid_stream_source_inline1660__tile
      ;
      uint64_t v195 = (uint64_t) v35;
      TASSIGN(v194, v195);
      // pto: %68
      ;
      int64_t v196 = v189 < v80 ? v80 : v189;
      // pto: %69
      ;
      int64_t v197 = v188 < v80 ? v80 : v188;
      // pto: %x_flat_inline1647__ssa_v0_pview
      ;
      const int64_t v198 = 0;
      // pto: %x_flat_inline1647__ssa_v0_pview
      ;
      __gm__ bfloat16_t* v199 = PTOAS__GLOBAL_TENSOR_DATA(v115);
      // pto: %x_flat_inline1647__ssa_v0_pview
      ;
      const int64_t v200 = 1;
      // pto: %x_flat_inline1647__ssa_v0_pview
      ;
      const int64_t v201 = 1;
      // pto: %x_flat_inline1647__ssa_v0_pview
      ;
      const int64_t v202 = 1;
      // pto: %x_flat_inline1647__ssa_v0_pview
      ;
      int64_t v203 = v192 * v71;
      // pto: %x_flat_inline1647__ssa_v0_pview
      ;
      int64_t v204 = v202 * v203;
      // pto: %x_flat_inline1647__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, -1> v205 = pto::Shape<1, 1, 1, -1, -1>(v200, v201, v202, v192, v83);
      // pto: %x_flat_inline1647__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v206 = pto::Stride<-1, -1, -1, -1, -1>(v201 * v204, v204, v203, v71, v69);
      // pto: %x_flat_inline1647__ssa_v0_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v207 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v199 + (v198 + v196 * v71 + v197 * v69), v205, v206);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v194, v207);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %0
      ;
      Tile<TileType::Vec, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v208 = Tile<TileType::Vec, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v192, v83);
      // pto: %0
      ;
      uint64_t v209 = (uint64_t) v36;
      TASSIGN(v208, v209);
      // pto: %71
      ;
      int64_t v210 = v193 < v80 ? v80 : v193;
      // pto: %72
      ;
      const int64_t v211 = 0;
      // pto: %72
      ;
      __gm__ bfloat16_t* v212 = PTOAS__GLOBAL_TENSOR_DATA(v115);
      // pto: %72
      ;
      const int64_t v213 = 1;
      // pto: %72
      ;
      const int64_t v214 = 1;
      // pto: %72
      ;
      const int64_t v215 = 1;
      // pto: %72
      ;
      int64_t v216 = v192 * v71;
      // pto: %72
      ;
      int64_t v217 = v215 * v216;
      // pto: %72
      ;
      pto::Shape<1, 1, 1, -1, -1> v218 = pto::Shape<1, 1, 1, -1, -1>(v213, v214, v215, v192, v83);
      // pto: %72
      ;
      pto::Stride<-1, -1, -1, -1, -1> v219 = pto::Stride<-1, -1, -1, -1, -1>(v214 * v217, v217, v216, v71, v69);
      // pto: %72
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v220 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v212 + (v211 + v196 * v71 + v210 * v69), v218, v219);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      TLOAD(v208, v220);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %mid_stream_widened_inline1655__tile
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v221 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v192, v83);
      // pto: %mid_stream_widened_inline1655__tile
      ;
      uint64_t v222 = (uint64_t) v37;
      TASSIGN(v221, v222);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      RoundMode v223 = RoundMode::CAST_ROUND;
      SaturationMode v224 = SaturationMode::ON;
      TCVT(v221, v194, v223, v224);
      v221.SetValidShape(v192, v83);
      // pto: %mid_stream_clean_inline1654__tile
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v225 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v68, v83);
      // pto: %mid_stream_clean_inline1654__tile
      ;
      uint64_t v226 = (uint64_t) v37;
      TASSIGN(v225, v226);
      pipe_barrier(PIPE_V);
      TFILLPAD<pto::TFillPadMode::InPlace>(v225, v221);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      v225.SetValidShape(v68, v83);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TPUSH<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v178, v225);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      // pto: %1
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v227 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v68, v83);
      // pto: %1
      ;
      uint64_t v228 = (uint64_t) v37;
      TASSIGN(v227, v228);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      TMUL(v227, v225, v225);
      // pto: %tmp_tile
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v229 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v83);
      // pto: %tmp_tile
      ;
      uint64_t v230 = (uint64_t) v35;
      TASSIGN(v229, v230);
      // pto: %2
      ;
      Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v231 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v69);
      // pto: %2
      ;
      uint64_t v232 = (uint64_t) v38;
      TASSIGN(v231, v232);
      pipe_barrier(PIPE_V);
      TROWSUM(v231, v227, v229);
      // pto: %mid_stream_sq_inline1852__tile
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v233 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v68);
      // pto: %mid_stream_sq_inline1852__tile
      ;
      uint64_t v234 = (uint64_t) v38;
      TASSIGN(v233, v234);
      // pto: %3
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v235 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v68);
      // pto: %3
      ;
      uint64_t v236 = (uint64_t) v38;
      TASSIGN(v235, v236);
      pipe_barrier(PIPE_V);
      TADD(v235, v184, v233);
      // pto: %4
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v237 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v192, v83);
      // pto: %4
      ;
      uint64_t v238 = (uint64_t) v37;
      TASSIGN(v237, v238);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      RoundMode v239 = RoundMode::CAST_ROUND;
      SaturationMode v240 = SaturationMode::ON;
      TCVT(v237, v208, v239, v240);
      v237.SetValidShape(v192, v83);
      // pto: %6
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v241 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v68, v83);
      // pto: %6
      ;
      uint64_t v242 = (uint64_t) v37;
      TASSIGN(v241, v242);
      pipe_barrier(PIPE_V);
      TFILLPAD<pto::TFillPadMode::InPlace>(v241, v237);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      v241.SetValidShape(v68, v83);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      TPUSH<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v178, v241);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      // pto: %8
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v243 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v68, v83);
      // pto: %8
      ;
      uint64_t v244 = (uint64_t) v37;
      TASSIGN(v243, v244);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      TMUL(v243, v241, v241);
      // pto: %9
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v245 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v83);
      // pto: %9
      ;
      uint64_t v246 = (uint64_t) v35;
      TASSIGN(v245, v246);
      // pto: %10
      ;
      Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v247 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v69);
      // pto: %10
      ;
      uint64_t v248 = (uint64_t) v36;
      TASSIGN(v247, v248);
      pipe_barrier(PIPE_V);
      TROWSUM(v247, v243, v245);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      // pto: %11
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v249 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v68);
      // pto: %11
      ;
      uint64_t v250 = (uint64_t) v36;
      TASSIGN(v249, v250);
      // pto: %12
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v251 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v68);
      // pto: %12
      ;
      uint64_t v252 = (uint64_t) v34;
      TASSIGN(v251, v252);
      pipe_barrier(PIPE_V);
      TADD(v251, v235, v249);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    };
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    // pto: %50, %51, %74, %rms_lane_inline1743__ssa_v0, %73
    ;
    int64_t v253 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v182 * (uint64_t) v16) + (uint64_t) v181) + (uint64_t) ((int64_t) (uint64_t) ((int64_t) v33) * (uint64_t) v68));
    // pto: %13
    ;
    Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v254 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v69);
    // pto: %13
    ;
    uint64_t v255 = (uint64_t) v34;
    TASSIGN(v254, v255);
    // pto: %75
    ;
    int64_t v256 = v253 < v80 ? v80 : v253;
    // pto: %rms_partials_inline1751__iter_v1_pview
    ;
    const int64_t v257 = 0;
    // pto: %rms_partials_inline1751__iter_v1_pview
    ;
    __gm__ float* v258 = PTOAS__GLOBAL_TENSOR_DATA(v106);
    // pto: %rms_partials_inline1751__iter_v1_pview
    ;
    const int64_t v259 = 1;
    // pto: %rms_partials_inline1751__iter_v1_pview
    ;
    const int64_t v260 = 1;
    // pto: %rms_partials_inline1751__iter_v1_pview
    ;
    const int64_t v261 = 1;
    // pto: %rms_partials_inline1751__iter_v1_pview
    ;
    int64_t v262 = v68 * v69;
    // pto: %rms_partials_inline1751__iter_v1_pview
    ;
    int64_t v263 = v261 * v262;
    // pto: %rms_partials_inline1751__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 32, 1> v264 = pto::Shape<1, 1, 1, 32, 1>(v259, v260, v261, v68, v69);
    // pto: %rms_partials_inline1751__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v265 = pto::Stride<-1, -1, -1, -1, -1>(v260 * v263, v263, v262, v69, v98);
    // pto: %rms_partials_inline1751__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v266 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v258 + (v257 + v256 * v69 + v80 * v98), v264, v265);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    TSTORE(v266, v254);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  pipe_barrier(PIPE_ALL);
  dsb(DSB_DDR);
  SYNCALL<SyncCoreType::Mix>();
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  // pto: %76, %77, %tail_lane_inline1775__ssa_v0
  int64_t v267 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v179 * (uint64_t) v82) + (uint64_t) ((int64_t) v33));
  // pto: %78
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  if (v267 < v18) {
    // pto: %80
    ;
    int64_t v268 = v267 < v20 ? v267 : v20;
    // pto: %79, %81
    ;
    int64_t v269 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v267 * (uint64_t) v19) + (uint64_t) v268);
    // pto: %83
    ;
    int64_t v270 = (int64_t) ((uint64_t) v20 - (uint64_t) v267);
    // pto: %84
    ;
    int64_t v271 = v270 < v80 ? v80 : v270;
    // pto: %85
    ;
    int64_t v272 = v271 < v69 ? v271 : v69;
    // pto: %82, %86
    ;
    int64_t v273 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v269 + (uint64_t) v19) + (uint64_t) v272);
    for (int64_t v274 = v269; v274 < v273; v274 += v73) {
      // pto: %scale2_inline1703__tile
      ;
      float v275 = (v5)[v82];
      // pto: %87
      ;
      int64_t v276 = (int64_t) ((uint64_t) v273 - (uint64_t) v274);
      // pto: %88
      ;
      int64_t v277 = v276 < v73 ? v276 : v73;
      // pto: %89
      ;
      int64_t v278 = (int64_t) ((uint64_t) v277 * (uint64_t) v68);
      // pto: %c_parts_load_inline1639__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v279 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v278);
      // pto: %c_parts_load_inline1639__ssa_v0
      ;
      uint64_t v280 = (uint64_t) v37;
      TASSIGN(v279, v280);
      // pto: %90
      ;
      int64_t v281 = (int64_t) ((uint64_t) v274 * (uint64_t) v68);
      // pto: %91
      ;
      int64_t v282 = v281 < v80 ? v80 : v281;
      // pto: %parts_2d_inline1805__ssa_v0_pview
      ;
      const int64_t v283 = 0;
      // pto: %parts_2d_inline1805__ssa_v0_pview
      ;
      __gm__ float* v284 = PTOAS__GLOBAL_TENSOR_DATA(v124);
      // pto: %parts_2d_inline1805__ssa_v0_pview
      ;
      const int64_t v285 = 1;
      // pto: %parts_2d_inline1805__ssa_v0_pview
      ;
      const int64_t v286 = 1;
      // pto: %parts_2d_inline1805__ssa_v0_pview
      ;
      const int64_t v287 = 1;
      // pto: %parts_2d_inline1805__ssa_v0_pview
      ;
      int64_t v288 = v73 * v116;
      // pto: %parts_2d_inline1805__ssa_v0_pview
      ;
      int64_t v289 = v287 * v288;
      // pto: %parts_2d_inline1805__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, -1> v290 = pto::Shape<1, 1, 1, 8, -1>(v285, v286, v287, v73, v278);
      // pto: %parts_2d_inline1805__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v291 = pto::Stride<-1, -1, -1, -1, -1>(v286 * v289, v289, v288, v116, v69);
      // pto: %parts_2d_inline1805__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v292 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v284 + (v283 + v80 * v116 + v282 * v69), v290, v291);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      TLOAD(v279, v292);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %t__tmp_v401
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v293 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v73, v84);
      // pto: %t__tmp_v401
      ;
      uint64_t v294 = (uint64_t) v37;
      TASSIGN(v293, v294);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      TFILLPAD<pto::TFillPadMode::InPlace>(v293, v279);
      v293.SetValidShape(v73, v84);
      // pto: %t__tmp_v402
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v295 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v84);
      // pto: %t__tmp_v402
      ;
      uint64_t v296 = (uint64_t) v37;
      TASSIGN(v295, v296);
      pipe_barrier(PIPE_V);
      TCOLSUM(v295, v293);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      // pto: %c_total_inline1630__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v297 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v68);
      // pto: %c_total_inline1630__ssa_v0
      ;
      uint64_t v298 = (uint64_t) v37;
      TASSIGN(v297, v298);
      // pto: %c_sq_load_inline1629__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v299 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v277);
      // pto: %c_sq_load_inline1629__ssa_v0
      ;
      uint64_t v300 = (uint64_t) v39;
      TASSIGN(v299, v300);
      // pto: %93
      ;
      int64_t v301 = v274 < v80 ? v80 : v274;
      // pto: %rms_2d_inline1755__ssa_v0_pview
      ;
      int64_t v302 = (int64_t) v29;
      // pto: %rms_2d_inline1755__ssa_v0_pview
      ;
      const int64_t v303 = 0;
      // pto: %rms_2d_inline1755__ssa_v0_pview
      ;
      __gm__ float* v304 = PTOAS__GLOBAL_TENSOR_DATA(v133);
      // pto: %rms_2d_inline1755__ssa_v0_pview
      ;
      const int64_t v305 = 1;
      // pto: %rms_2d_inline1755__ssa_v0_pview
      ;
      const int64_t v306 = 1;
      // pto: %rms_2d_inline1755__ssa_v0_pview
      ;
      const int64_t v307 = 1;
      // pto: %rms_2d_inline1755__ssa_v0_pview
      ;
      int64_t v308 = v73 * v302;
      // pto: %rms_2d_inline1755__ssa_v0_pview
      ;
      int64_t v309 = v307 * v308;
      // pto: %rms_2d_inline1755__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, -1> v310 = pto::Shape<1, 1, 1, 8, -1>(v305, v306, v307, v73, v277);
      // pto: %rms_2d_inline1755__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v311 = pto::Stride<-1, -1, -1, -1, -1>(v306 * v309, v309, v308, v302, v69);
      // pto: %rms_2d_inline1755__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v312 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v304 + (v303 + v80 * v302 + v301 * v69), v310, v311);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      TLOAD(v299, v312);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      // pto: %t__tmp_v403
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v313 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v73, v73);
      // pto: %t__tmp_v403
      ;
      uint64_t v314 = (uint64_t) v39;
      TASSIGN(v313, v314);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      TFILLPAD<pto::TFillPadMode::InPlace>(v313, v299);
      v313.SetValidShape(v73, v73);
      // pto: %c_rms_tmp_inline1665__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v315 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v73);
      // pto: %c_rms_tmp_inline1665__ssa_v0
      ;
      uint64_t v316 = (uint64_t) v40;
      TASSIGN(v315, v316);
      // pto: %t__tmp_v404
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v317 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v73);
      // pto: %t__tmp_v404
      ;
      uint64_t v318 = (uint64_t) v39;
      TASSIGN(v317, v318);
      pipe_barrier(PIPE_V);
      TCOLSUM(v317, v313);
      // pto: %t__tmp_v405
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v319 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v73);
      // pto: %t__tmp_v405
      ;
      uint64_t v320 = (uint64_t) v39;
      TASSIGN(v319, v320);
      pipe_barrier(PIPE_V);
      TMULS(v319, v317, v85);
      // pto: %t__tmp_v406
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v321 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v73);
      // pto: %t__tmp_v406
      ;
      uint64_t v322 = (uint64_t) v39;
      TASSIGN(v321, v322);
      pipe_barrier(PIPE_V);
      TADDS(v321, v319, v86);
      // pto: %t__tmp_v407
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v323 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v73);
      // pto: %t__tmp_v407
      ;
      uint64_t v324 = (uint64_t) v41;
      TASSIGN(v323, v324);
      pipe_barrier(PIPE_V);
      TRSQRT(v323, v321, v315);
      // pto: %c_inv_inline1723__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v325 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v69);
      // pto: %c_inv_inline1723__ssa_v0
      ;
      uint64_t v326 = (uint64_t) v41;
      TASSIGN(v325, v326);
      // pto: %comb_inv_inline1640__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v327 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v69);
      // pto: %comb_inv_inline1640__ssa_v0
      ;
      uint64_t v328 = (uint64_t) v41;
      TASSIGN(v327, v328);
      // pto: %t__tmp_v408
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v329 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v74);
      // pto: %t__tmp_v408
      ;
      uint64_t v330 = (uint64_t) v42;
      TASSIGN(v329, v330);
      // pto: %slice_view
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 16, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v331;
      // pto: %slice_view
      ;
      uint64_t v332 = (uint64_t) v42;
      TASSIGN(v331, v332);
      // pto: %raw_inline1627__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v333 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v74);
      // pto: %raw_inline1627__ssa_v0
      ;
      uint64_t v334 = (uint64_t) v37;
      TASSIGN(v333, v334);
      TADDS(v333, v331, v79);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      // pto: %base_inline1786__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v335 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v74);
      // pto: %base_inline1786__ssa_v0
      ;
      uint64_t v336 = (uint64_t) v43;
      TASSIGN(v335, v336);
      // pto: %hc_base_2d_inline1697__ssa_v0_pview
      ;
      __gm__ float* v337 = PTOAS__GLOBAL_TENSOR_DATA(v141);
      // pto: %hc_base_2d_inline1697__ssa_v0_pview
      ;
      unsigned v338 = 8;
      // pto: %hc_base_2d_inline1697__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 16> v339 = pto::Shape<1, 1, 1, 1, 16>();
      // pto: %hc_base_2d_inline1697__ssa_v0_pview
      ;
      pto::Stride<24, 24, 24, 24, 1> v340 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %hc_base_2d_inline1697__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v341 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v337 + v338, v339, v340);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      TLOAD(v335, v341);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      // pto: %t__tmp_v409
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v342 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v74);
      // pto: %t__tmp_v409
      ;
      uint64_t v343 = (uint64_t) v37;
      TASSIGN(v342, v343);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v342, v333, v327);
      // pto: %t__tmp_v410
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v344 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v74);
      // pto: %t__tmp_v410
      ;
      uint64_t v345 = (uint64_t) v37;
      TASSIGN(v344, v345);
      pipe_barrier(PIPE_V);
      TMULS(v344, v342, v275);
      // pto: %logits_inline1817__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v346 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v74);
      // pto: %logits_inline1817__ssa_v0
      ;
      uint64_t v347 = (uint64_t) v44;
      TASSIGN(v346, v347);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      TCOLEXPANDADD(v346, v344, v335);
      // pto: %t__tmp_v411
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v348 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v74);
      // pto: %t__tmp_v411
      ;
      uint64_t v349 = (uint64_t) v44;
      TASSIGN(v348, v349);
      // pto: %regular_logits_inline1724__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v350 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %regular_logits_inline1724__ssa_v0
      ;
      uint64_t v351 = (uint64_t) v44;
      TASSIGN(v350, v351);
      // pto: %ids_inline1712__ci_tmp_v0
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v352 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v88);
      // pto: %ids_inline1712__ci_tmp_v0
      ;
      uint64_t v353 = (uint64_t) v37;
      TASSIGN(v352, v353);
      // pto: %ids_inline1712__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v354 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %ids_inline1712__ssa_v1
      ;
      uint64_t v355 = (uint64_t) v40;
      TASSIGN(v354, v355);
      // pto: %ci_tmp_view
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v356;
      TRESHAPE(v356, v352);
      // pto: %ci_dst_view
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v357;
      TRESHAPE(v357, v354);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v357, v76, v356);
      // pto: %tmp_inline1860__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v358 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %tmp_inline1860__ssa_v1
      ;
      uint64_t v359 = (uint64_t) v37;
      TASSIGN(v358, v359);
      // pto: %ids16_inline1682__ssa_v1
      ;
      Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v360 = Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v84);
      // pto: %ids16_inline1682__ssa_v1
      ;
      uint64_t v361 = (uint64_t) v40;
      TASSIGN(v360, v361);
      // pto: %xor_tmp16_inline1797__ssa_v1
      ;
      Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v362 = Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v84);
      // pto: %xor_tmp16_inline1797__ssa_v1
      ;
      uint64_t v363 = (uint64_t) v45;
      TASSIGN(v362, v363);
      // pto: %t__tmp_v412
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v364 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %t__tmp_v412
      ;
      uint64_t v365 = (uint64_t) v46;
      TASSIGN(v364, v365);
      TEXPANDS(v364, v89);
      // pto: %ri1_mask_inline1807__ssa_v1
      ;
      Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v366 = Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v84);
      // pto: %ri1_mask_inline1807__ssa_v1
      ;
      uint64_t v367 = (uint64_t) v46;
      TASSIGN(v366, v367);
      // pto: %t__tmp_v413
      ;
      Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v368 = Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v84);
      // pto: %t__tmp_v413
      ;
      uint64_t v369 = (uint64_t) v47;
      TASSIGN(v368, v369);
      pipe_barrier(PIPE_V);
      TXOR(v368, v360, v366, v362);
      // pto: %ri1_inline1810__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v370 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %ri1_inline1810__ssa_v1
      ;
      uint64_t v371 = (uint64_t) v47;
      TASSIGN(v370, v371);
      // pto: %t__tmp_v414
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v372 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %t__tmp_v414
      ;
      uint64_t v373 = (uint64_t) v46;
      TASSIGN(v372, v373);
      pipe_barrier(PIPE_V);
      TEXPANDS(v372, v90);
      // pto: %ri2_mask_inline1637__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v374 = Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v84);
      // pto: %ri2_mask_inline1637__ssa_v0
      ;
      uint64_t v375 = (uint64_t) v46;
      TASSIGN(v374, v375);
      // pto: %t__tmp_v415
      ;
      Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v376 = Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v84);
      // pto: %t__tmp_v415
      ;
      uint64_t v377 = (uint64_t) v48;
      TASSIGN(v376, v377);
      pipe_barrier(PIPE_V);
      TXOR(v376, v360, v374, v362);
      // pto: %ri2_inline1814__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v378 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %ri2_inline1814__ssa_v0
      ;
      uint64_t v379 = (uint64_t) v48;
      TASSIGN(v378, v379);
      // pto: %t__tmp_v416
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v380 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %t__tmp_v416
      ;
      uint64_t v381 = (uint64_t) v46;
      TASSIGN(v380, v381);
      pipe_barrier(PIPE_V);
      TEXPANDS(v380, v91);
      // pto: %ci1_mask_inline1833__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v382 = Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v84);
      // pto: %ci1_mask_inline1833__ssa_v0
      ;
      uint64_t v383 = (uint64_t) v46;
      TASSIGN(v382, v383);
      // pto: %t__tmp_v417
      ;
      Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v384 = Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v84);
      // pto: %t__tmp_v417
      ;
      uint64_t v385 = (uint64_t) v49;
      TASSIGN(v384, v385);
      pipe_barrier(PIPE_V);
      TXOR(v384, v360, v382, v362);
      // pto: %ci1_inline1791__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v386 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %ci1_inline1791__ssa_v0
      ;
      uint64_t v387 = (uint64_t) v49;
      TASSIGN(v386, v387);
      // pto: %t__tmp_v418
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v388 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %t__tmp_v418
      ;
      uint64_t v389 = (uint64_t) v46;
      TASSIGN(v388, v389);
      pipe_barrier(PIPE_V);
      TEXPANDS(v388, v92);
      // pto: %ci2_mask_inline1835__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v390 = Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v84);
      // pto: %ci2_mask_inline1835__ssa_v0
      ;
      uint64_t v391 = (uint64_t) v46;
      TASSIGN(v390, v391);
      // pto: %t__tmp_v419
      ;
      Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v392 = Tile<TileType::Vec, int16_t, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v84);
      // pto: %t__tmp_v419
      ;
      uint64_t v393 = (uint64_t) v50;
      TASSIGN(v392, v393);
      pipe_barrier(PIPE_V);
      TXOR(v392, v360, v390, v362);
      // pto: %ci2_inline1789__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v394 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %ci2_inline1789__ssa_v0
      ;
      uint64_t v395 = (uint64_t) v50;
      TASSIGN(v394, v395);
      // pto: %t__tmp_v420
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v396 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %t__tmp_v420
      ;
      uint64_t v397 = (uint64_t) v40;
      TASSIGN(v396, v397);
      pipe_barrier(PIPE_V);
      TGATHER(v396, v350, v370, v358);
      // pto: %max2_inline1734__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v398 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %max2_inline1734__ssa_v1
      ;
      uint64_t v399 = (uint64_t) v40;
      TASSIGN(v398, v399);
      pipe_barrier(PIPE_V);
      TMAX(v398, v350, v396);
      // pto: %t__tmp_v421
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v400 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %t__tmp_v421
      ;
      uint64_t v401 = (uint64_t) v45;
      TASSIGN(v400, v401);
      pipe_barrier(PIPE_V);
      TGATHER(v400, v398, v378, v358);
      // pto: %max4_inline1813__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v402 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %max4_inline1813__ssa_v1
      ;
      uint64_t v403 = (uint64_t) v40;
      TASSIGN(v402, v403);
      pipe_barrier(PIPE_V);
      TMAX(v402, v398, v400);
      // pto: %t__tmp_v422
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v404 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %t__tmp_v422
      ;
      uint64_t v405 = (uint64_t) v43;
      TASSIGN(v404, v405);
      pipe_barrier(PIPE_V);
      TSUB(v404, v350, v402);
      // pto: %exps_inline1651__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v406 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %exps_inline1651__ssa_v1
      ;
      uint64_t v407 = (uint64_t) v43;
      TASSIGN(v406, v407);
      pipe_barrier(PIPE_V);
      TEXP(v406, v404);
      // pto: %t__tmp_v423
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v408 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %t__tmp_v423
      ;
      uint64_t v409 = (uint64_t) v39;
      TASSIGN(v408, v409);
      pipe_barrier(PIPE_V);
      TGATHER(v408, v406, v370, v358);
      // pto: %sum2_inline1644__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v410 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %sum2_inline1644__ssa_v1
      ;
      uint64_t v411 = (uint64_t) v39;
      TASSIGN(v410, v411);
      pipe_barrier(PIPE_V);
      TADD(v410, v406, v408);
      // pto: %t__tmp_v424
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v412 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %t__tmp_v424
      ;
      uint64_t v413 = (uint64_t) v51;
      TASSIGN(v412, v413);
      pipe_barrier(PIPE_V);
      TGATHER(v412, v410, v378, v358);
      // pto: %sum4_inline1871__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v414 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %sum4_inline1871__ssa_v1
      ;
      uint64_t v415 = (uint64_t) v39;
      TASSIGN(v414, v415);
      pipe_barrier(PIPE_V);
      TADD(v414, v410, v412);
      // pto: %t__tmp_v425
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v416 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %t__tmp_v425
      ;
      uint64_t v417 = (uint64_t) v43;
      TASSIGN(v416, v417);
      pipe_barrier(PIPE_V);
      TDIV(v416, v406, v414);
      // pto: %probs_inline1824__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v418 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %probs_inline1824__ssa_v1
      ;
      uint64_t v419 = (uint64_t) v43;
      TASSIGN(v418, v419);
      pipe_barrier(PIPE_V);
      TADDS(v418, v416, v86);
      // pto: %t__tmp_v426
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v420 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %t__tmp_v426
      ;
      uint64_t v421 = (uint64_t) v39;
      TASSIGN(v420, v421);
      pipe_barrier(PIPE_V);
      TGATHER(v420, v418, v386, v358);
      // pto: %col2_inline1692__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v422 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %col2_inline1692__ssa_v4
      ;
      uint64_t v423 = (uint64_t) v39;
      TASSIGN(v422, v423);
      pipe_barrier(PIPE_V);
      TADD(v422, v418, v420);
      // pto: %t__tmp_v427
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v424 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %t__tmp_v427
      ;
      uint64_t v425 = (uint64_t) v51;
      TASSIGN(v424, v425);
      pipe_barrier(PIPE_V);
      TGATHER(v424, v422, v394, v358);
      // pto: %t__tmp_v428
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v426 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %t__tmp_v428
      ;
      uint64_t v427 = (uint64_t) v39;
      TASSIGN(v426, v427);
      pipe_barrier(PIPE_V);
      TADD(v426, v422, v424);
      // pto: %col4_inline1821__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v428 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %col4_inline1821__ssa_v4
      ;
      uint64_t v429 = (uint64_t) v39;
      TASSIGN(v428, v429);
      pipe_barrier(PIPE_V);
      TADDS(v428, v426, v86);
      // pto: %cur_inline1827__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v430 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
      // pto: %cur_inline1827__ssa_v4
      ;
      uint64_t v431 = (uint64_t) v43;
      TASSIGN(v430, v431);
      pipe_barrier(PIPE_V);
      TDIV(v430, v418, v428);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      for (int64_t v432 = v80; v432 < v93; v432 += v69) {
        // pto: %t__tmp_v429
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v433 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
        // pto: %t__tmp_v429
        ;
        uint64_t v434 = (uint64_t) v39;
        TASSIGN(v433, v434);
        pipe_barrier(PIPE_V);
        TGATHER(v433, v430, v370, v358);
        // pto: %row2_inline1829__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v435 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
        // pto: %row2_inline1829__ssa_v1
        ;
        uint64_t v436 = (uint64_t) v39;
        TASSIGN(v435, v436);
        pipe_barrier(PIPE_V);
        TADD(v435, v430, v433);
        // pto: %t__tmp_v430
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v437 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
        // pto: %t__tmp_v430
        ;
        uint64_t v438 = (uint64_t) v51;
        TASSIGN(v437, v438);
        pipe_barrier(PIPE_V);
        TGATHER(v437, v435, v378, v358);
        // pto: %t__tmp_v431
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v439 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
        // pto: %t__tmp_v431
        ;
        uint64_t v440 = (uint64_t) v39;
        TASSIGN(v439, v440);
        pipe_barrier(PIPE_V);
        TADD(v439, v435, v437);
        // pto: %row4_inline1882__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v441 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
        // pto: %row4_inline1882__ssa_v1
        ;
        uint64_t v442 = (uint64_t) v39;
        TASSIGN(v441, v442);
        pipe_barrier(PIPE_V);
        TADDS(v441, v439, v86);
        // pto: %norm_inline1846__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v443 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
        // pto: %norm_inline1846__ssa_v1
        ;
        uint64_t v444 = (uint64_t) v39;
        TASSIGN(v443, v444);
        pipe_barrier(PIPE_V);
        TDIV(v443, v430, v441);
        // pto: %t__tmp_v432
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v445 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
        // pto: %t__tmp_v432
        ;
        uint64_t v446 = (uint64_t) v51;
        TASSIGN(v445, v446);
        pipe_barrier(PIPE_V);
        TGATHER(v445, v443, v386, v358);
        // pto: %col2_inline1692__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v447 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
        // pto: %col2_inline1692__ssa_v7
        ;
        uint64_t v448 = (uint64_t) v51;
        TASSIGN(v447, v448);
        pipe_barrier(PIPE_V);
        TADD(v447, v443, v445);
        // pto: %t__tmp_v433
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v449 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
        // pto: %t__tmp_v433
        ;
        uint64_t v450 = (uint64_t) v52;
        TASSIGN(v449, v450);
        pipe_barrier(PIPE_V);
        TGATHER(v449, v447, v394, v358);
        // pto: %t__tmp_v434
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v451 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
        // pto: %t__tmp_v434
        ;
        uint64_t v452 = (uint64_t) v51;
        TASSIGN(v451, v452);
        pipe_barrier(PIPE_V);
        TADD(v451, v447, v449);
        // pto: %col4_inline1821__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v453 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
        // pto: %col4_inline1821__ssa_v7
        ;
        uint64_t v454 = (uint64_t) v51;
        TASSIGN(v453, v454);
        pipe_barrier(PIPE_V);
        TADDS(v453, v451, v86);
        // pto: %cur_inline1827__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v455 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v87);
        // pto: %cur_inline1827__ssa_v7
        ;
        uint64_t v456 = (uint64_t) v43;
        TASSIGN(v455, v456);
        pipe_barrier(PIPE_V);
        TDIV(v455, v443, v453);
      };
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      // pto: %t__tmp_v435
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v457 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v74);
      // pto: %t__tmp_v435
      ;
      uint64_t v458 = (uint64_t) v43;
      TASSIGN(v457, v458);
      v457.SetValidShape(v277, v74);
      // pto: %comb_inline1666__ssa_v0_pview
      ;
      const int64_t v459 = 0;
      // pto: %comb_inline1666__ssa_v0_pview
      ;
      __gm__ float* v460 = PTOAS__GLOBAL_TENSOR_DATA(v150);
      // pto: %comb_inline1666__ssa_v0_pview
      ;
      const int64_t v461 = 1;
      // pto: %comb_inline1666__ssa_v0_pview
      ;
      const int64_t v462 = 1;
      // pto: %comb_inline1666__ssa_v0_pview
      ;
      const int64_t v463 = 1;
      // pto: %comb_inline1666__ssa_v0_pview
      ;
      int64_t v464 = v277 * v74;
      // pto: %comb_inline1666__ssa_v0_pview
      ;
      int64_t v465 = v463 * v464;
      // pto: %comb_inline1666__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, 16> v466 = pto::Shape<1, 1, 1, -1, 16>(v461, v462, v463, v277, v74);
      // pto: %comb_inline1666__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v467 = pto::Stride<-1, -1, -1, -1, -1>(v462 * v465, v465, v464, v74, v69);
      // pto: %comb_inline1666__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v468 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v460 + (v459 + v301 * v74 + v80 * v69), v466, v467);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      TSTORE(v468, v457);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    };
  } else {
    // pto: %96
    ;
    int64_t v469 = (int64_t) ((uint64_t) v267 - (uint64_t) v18);
    // pto: %98
    ;
    int64_t v470 = v469 < v22 ? v469 : v22;
    // pto: %97, %99, %100
    ;
    int64_t v471 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v469 * (uint64_t) v21) + (uint64_t) v470) * (uint64_t) v82);
    // pto: %101
    ;
    int64_t v472 = (int64_t) ((uint64_t) v22 - (uint64_t) v469);
    // pto: %102
    ;
    int64_t v473 = v472 < v80 ? v80 : v472;
    // pto: %103
    ;
    int64_t v474 = v473 < v69 ? v473 : v69;
    // pto: %106, %104, %105
    ;
    int64_t v475 = (int64_t) ((uint64_t) v471 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v21 + (uint64_t) v474) * (uint64_t) v82));
    // pto: %107
    ;
    // pto: %108
    ;
    // pto: %d_begin_inline1804__phi_v2
    ;
    int64_t v476;
    // pto: %work_begin_inline1657__phi_v3
    ;
    int64_t v477;
    // pto: %work_end_inline1688__phi_v3
    ;
    int64_t v478;
    if (v24 < v25) {
      // pto: %109
      ;
      // pto: %d_worker_inline1889__phi_v2
      ;
      int64_t v479;
      // pto: %d_workers_inline1890__phi_v2
      ;
      int64_t v480;
      // pto: %pair_id_inline1744__phi_v2
      ;
      int64_t v481;
      if (v469 < v26) {
        // pto: %110
        ;
        int64_t v482 = (int64_t) ((uint64_t) v27 + (uint64_t) v69);
        // pto: %111
        ;
        // pto: %113
        ;
        v479 = v469 % v482;
        v480 = v482;
        v481 = v469 / v482;
      } else {
        // pto: %115
        ;
        int64_t v483 = (int64_t) ((uint64_t) v469 - (uint64_t) v26);
        // pto: %117, %116
        ;
        // pto: %118
        ;
        v479 = v483 % v27;
        v480 = v27;
        v481 = (int64_t) ((uint64_t) v28 + (uint64_t) (v483 / v27));
      };
      // pto: %119
      ;
      uint64_t v484 = (uint64_t) v481;
      // pto: %119
      ;
      int64_t v485 = (int64_t) (v484 * (uint64_t) v82);
      // pto: %120
      ;
      int64_t v486 = v23 < v485 ? v23 : v485;
      // pto: %121
      ;
      int64_t v487 = (int64_t) ((uint64_t) v486 + (uint64_t) v82);
      // pto: %122
      ;
      // pto: %123
      ;
      int64_t v488 = v73 / v480;
      // pto: %124
      ;
      int64_t v489 = v73 % v480;
      // pto: %125
      ;
      uint64_t v490 = (uint64_t) v479;
      // pto: %126
      ;
      bool v491 = v479 < v489;
      // pto: %126
      ;
      int64_t v492 = v491 ? v479 : v489;
      // pto: %125, %127
      ;
      v476 = (int64_t) ((uint64_t) ((int64_t) v490 * (uint64_t) v488) + (uint64_t) v492);
      v477 = v486;
      v478 = v23 < v487 ? v23 : v487;
    } else {
      v476 = v80;
      v477 = v471;
      v478 = v23 < v475 ? v23 : v475;
    };
    // pto: %128
    ;
    bool v493 = v25 <= v469;
    // pto: %work_begin_inline1657__phi_v5
    ;
    int64_t v494 = v493 ? v80 : v477;
    // pto: %work_end_inline1688__phi_v5
    ;
    int64_t v495 = v493 ? v80 : v478;
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    for (int64_t v496 = v494; v496 < v495; v496 += v73) {
      // pto: %129
      ;
      int64_t v497 = (int64_t) ((uint64_t) v495 - (uint64_t) v496);
      // pto: %130
      ;
      int64_t v498 = v497 < v73 ? v497 : v73;
      // pto: %131
      ;
      int64_t v499 = (int64_t) ((uint64_t) v498 * (uint64_t) v68);
      // pto: %m_parts_load_inline1897__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v500 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v499);
      // pto: %m_parts_load_inline1897__ssa_v0
      ;
      uint64_t v501 = (uint64_t) v37;
      TASSIGN(v500, v501);
      // pto: %132
      ;
      int64_t v502 = (int64_t) ((uint64_t) v496 * (uint64_t) v68);
      // pto: %133
      ;
      int64_t v503 = v502 < v80 ? v80 : v502;
      // pto: %135
      ;
      const int64_t v504 = 0;
      // pto: %135
      ;
      __gm__ float* v505 = PTOAS__GLOBAL_TENSOR_DATA(v124);
      // pto: %135
      ;
      const int64_t v506 = 1;
      // pto: %135
      ;
      const int64_t v507 = 1;
      // pto: %135
      ;
      const int64_t v508 = 1;
      // pto: %135
      ;
      int64_t v509 = v73 * v116;
      // pto: %135
      ;
      int64_t v510 = v508 * v509;
      // pto: %135
      ;
      pto::Shape<1, 1, 1, 8, -1> v511 = pto::Shape<1, 1, 1, 8, -1>(v506, v507, v508, v73, v499);
      // pto: %135
      ;
      pto::Stride<-1, -1, -1, -1, -1> v512 = pto::Stride<-1, -1, -1, -1, -1>(v507 * v510, v510, v509, v116, v69);
      // pto: %135
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v513 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v505 + (v504 + v80 * v116 + v503 * v69), v511, v512);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      TLOAD(v500, v513);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      // pto: %t__tmp_v436
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v514 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v73, v84);
      // pto: %t__tmp_v436
      ;
      uint64_t v515 = (uint64_t) v37;
      TASSIGN(v514, v515);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      TFILLPAD<pto::TFillPadMode::InPlace>(v514, v500);
      v514.SetValidShape(v73, v84);
      // pto: %t__tmp_v437
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v516 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v84);
      // pto: %t__tmp_v437
      ;
      uint64_t v517 = (uint64_t) v37;
      TASSIGN(v516, v517);
      pipe_barrier(PIPE_V);
      TCOLSUM(v516, v514);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      // pto: %m_total_inline1899__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v518 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v68);
      // pto: %m_total_inline1899__ssa_v0
      ;
      uint64_t v519 = (uint64_t) v37;
      TASSIGN(v518, v519);
      // pto: %m_sq_load_inline1847__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v520 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v498);
      // pto: %m_sq_load_inline1847__ssa_v0
      ;
      uint64_t v521 = (uint64_t) v39;
      TASSIGN(v520, v521);
      // pto: %136
      ;
      int64_t v522 = v496 < v80 ? v80 : v496;
      // pto: %137
      ;
      int64_t v523 = (int64_t) v29;
      // pto: %137
      ;
      const int64_t v524 = 0;
      // pto: %137
      ;
      __gm__ float* v525 = PTOAS__GLOBAL_TENSOR_DATA(v133);
      // pto: %137
      ;
      const int64_t v526 = 1;
      // pto: %137
      ;
      const int64_t v527 = 1;
      // pto: %137
      ;
      const int64_t v528 = 1;
      // pto: %137
      ;
      int64_t v529 = v73 * v523;
      // pto: %137
      ;
      int64_t v530 = v528 * v529;
      // pto: %137
      ;
      pto::Shape<1, 1, 1, 8, -1> v531 = pto::Shape<1, 1, 1, 8, -1>(v526, v527, v528, v73, v498);
      // pto: %137
      ;
      pto::Stride<-1, -1, -1, -1, -1> v532 = pto::Stride<-1, -1, -1, -1, -1>(v527 * v530, v530, v529, v523, v69);
      // pto: %137
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v533 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v525 + (v524 + v80 * v523 + v522 * v69), v531, v532);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      TLOAD(v520, v533);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      // pto: %t__tmp_v438
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v534 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v73, v73);
      // pto: %t__tmp_v438
      ;
      uint64_t v535 = (uint64_t) v39;
      TASSIGN(v534, v535);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      TFILLPAD<pto::TFillPadMode::InPlace>(v534, v520);
      v534.SetValidShape(v73, v73);
      // pto: %m_rms_tmp_inline1799__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v536 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v73);
      // pto: %m_rms_tmp_inline1799__ssa_v0
      ;
      uint64_t v537 = (uint64_t) v40;
      TASSIGN(v536, v537);
      // pto: %t__tmp_v439
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v538 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v73);
      // pto: %t__tmp_v439
      ;
      uint64_t v539 = (uint64_t) v39;
      TASSIGN(v538, v539);
      pipe_barrier(PIPE_V);
      TCOLSUM(v538, v534);
      // pto: %t__tmp_v440
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v540 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v73);
      // pto: %t__tmp_v440
      ;
      uint64_t v541 = (uint64_t) v39;
      TASSIGN(v540, v541);
      pipe_barrier(PIPE_V);
      TMULS(v540, v538, v85);
      // pto: %t__tmp_v441
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v542 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v73);
      // pto: %t__tmp_v441
      ;
      uint64_t v543 = (uint64_t) v39;
      TASSIGN(v542, v543);
      pipe_barrier(PIPE_V);
      TADDS(v542, v540, v86);
      // pto: %t__tmp_v442
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v544 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v73);
      // pto: %t__tmp_v442
      ;
      uint64_t v545 = (uint64_t) v45;
      TASSIGN(v544, v545);
      pipe_barrier(PIPE_V);
      TRSQRT(v544, v542, v536);
      // pto: %m_inv_inline1900__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v546 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v69);
      // pto: %m_inv_inline1900__ssa_v0
      ;
      uint64_t v547 = (uint64_t) v45;
      TASSIGN(v546, v547);
      // pto: %gate_base_inline1831__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v548 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v73);
      // pto: %gate_base_inline1831__ssa_v0
      ;
      uint64_t v549 = (uint64_t) v53;
      TASSIGN(v548, v549);
      // pto: %138
      ;
      __gm__ float* v550 = PTOAS__GLOBAL_TENSOR_DATA(v141);
      // pto: %138
      ;
      pto::Shape<1, 1, 1, 1, 8> v551 = pto::Shape<1, 1, 1, 1, 8>();
      // pto: %138
      ;
      pto::Stride<24, 24, 24, 24, 1> v552 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %138
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v553 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v550, v551, v552);
      TLOAD(v548, v553);
      // pto: %gate_scales_inline1634__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v554 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v72);
      // pto: %gate_scales_inline1634__ssa_v0
      ;
      uint64_t v555 = (uint64_t) v54;
      TASSIGN(v554, v555);
      // pto: %hc_scale_2d_inline1720__ssa_v0_pview
      ;
      __gm__ float* v556 = PTOAS__GLOBAL_TENSOR_DATA(v158);
      // pto: %hc_scale_2d_inline1720__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 3> v557 = pto::Shape<1, 1, 1, 1, 3>();
      // pto: %hc_scale_2d_inline1720__ssa_v0_pview
      ;
      pto::Stride<3, 3, 3, 3, 1> v558 = pto::Stride<3, 3, 3, 3, 1>();
      // pto: %hc_scale_2d_inline1720__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND> v559 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND>(v556, v557, v558);
      TLOAD(v554, v559);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
      // pto: %t__ci_tmp_v1
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v560 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v88);
      // pto: %t__ci_tmp_v1
      ;
      uint64_t v561 = (uint64_t) v39;
      TASSIGN(v560, v561);
      // pto: %t__tmp_v443
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v562 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v73);
      // pto: %t__tmp_v443
      ;
      uint64_t v563 = (uint64_t) v55;
      TASSIGN(v562, v563);
      // pto: %139
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v564;
      TRESHAPE(v564, v560);
      // pto: %140
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v565;
      TRESHAPE(v565, v562);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v565, v76, v564);
      // pto: %gate_ids_inline1901__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v566 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v73);
      // pto: %gate_ids_inline1901__ssa_v0
      ;
      uint64_t v567 = (uint64_t) v39;
      TASSIGN(v566, v567);
      pipe_barrier(PIPE_V);
      RoundMode v568 = RoundMode::CAST_ROUND;
      SaturationMode v569 = SaturationMode::ON;
      TCVT(v566, v562, v568, v569);
      // pto: %t__tmp_v444
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v570 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v73);
      // pto: %t__tmp_v444
      ;
      uint64_t v571 = (uint64_t) v39;
      TASSIGN(v570, v571);
      pipe_barrier(PIPE_V);
      TMULS(v570, v566, v94);
      // pto: %gate_scale_ids_inline1902__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v572 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v73);
      // pto: %gate_scale_ids_inline1902__ssa_v0
      ;
      uint64_t v573 = (uint64_t) v39;
      TASSIGN(v572, v573);
      pipe_barrier(PIPE_V);
      RoundMode v574 = RoundMode::CAST_FLOOR;
      SaturationMode v575 = SaturationMode::ON;
      TCVT(v572, v570, v574, v575);
      // pto: %gate_tmp_inline1795__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v576 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v73);
      // pto: %gate_tmp_inline1795__ssa_v0
      ;
      uint64_t v577 = (uint64_t) v41;
      TASSIGN(v576, v577);
      // pto: %gate_scale_vec_inline1904__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v578 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v73);
      // pto: %gate_scale_vec_inline1904__ssa_v0
      ;
      uint64_t v579 = (uint64_t) v56;
      TASSIGN(v578, v579);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
      TGATHER(v578, v554, v572, v576);
      // pto: %t__tmp_v445
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v580 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v73);
      // pto: %t__tmp_v445
      ;
      uint64_t v581 = (uint64_t) v37;
      TASSIGN(v580, v581);
      // pto: %141
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v582;
      // pto: %141
      ;
      uint64_t v583 = (uint64_t) v37;
      TASSIGN(v582, v583);
      // pto: %gate_raw_inline1905__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v584 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v73);
      // pto: %gate_raw_inline1905__ssa_v0
      ;
      uint64_t v585 = (uint64_t) v37;
      TASSIGN(v584, v585);
      TADDS(v584, v582, v79);
      // pto: %gate_norm_inline1888__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v586 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v73);
      // pto: %gate_norm_inline1888__ssa_v0
      ;
      uint64_t v587 = (uint64_t) v37;
      TASSIGN(v586, v587);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v586, v584, v546);
      // pto: %gate_scaled_inline1754__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v588 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v73);
      // pto: %gate_scaled_inline1754__ssa_v0
      ;
      uint64_t v589 = (uint64_t) v37;
      TASSIGN(v588, v589);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v588, v586, v578);
      // pto: %t__tmp_v446
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v590 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v73);
      // pto: %t__tmp_v446
      ;
      uint64_t v591 = (uint64_t) v57;
      TASSIGN(v590, v591);
      TCOLEXPAND(v590, v548);
      // pto: %gate_logits_inline1903__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v592 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v73);
      // pto: %gate_logits_inline1903__ssa_v0
      ;
      uint64_t v593 = (uint64_t) v37;
      TASSIGN(v592, v593);
      pipe_barrier(PIPE_V);
      TADD(v592, v588, v590);
      // pto: %t__tmp_v447
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v594 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v73);
      // pto: %t__tmp_v447
      ;
      uint64_t v595 = (uint64_t) v37;
      TASSIGN(v594, v595);
      pipe_barrier(PIPE_V);
      TNEG(v594, v592);
      // pto: %t__tmp_v448
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v596 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v73);
      // pto: %t__tmp_v448
      ;
      uint64_t v597 = (uint64_t) v37;
      TASSIGN(v596, v597);
      pipe_barrier(PIPE_V);
      TEXP(v596, v594);
      // pto: %t__tmp_v449
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v598 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v73);
      // pto: %t__tmp_v449
      ;
      uint64_t v599 = (uint64_t) v37;
      TASSIGN(v598, v599);
      pipe_barrier(PIPE_V);
      TADDS(v598, v596, v95);
      // pto: %gate_sigmoid_inline1906__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v600 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v73);
      // pto: %gate_sigmoid_inline1906__ssa_v0
      ;
      uint64_t v601 = (uint64_t) v58;
      TASSIGN(v600, v601);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
      TRECIP(v600, v598);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      // pto: %t__ci_tmp_v2
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v602 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v88);
      // pto: %t__ci_tmp_v2
      ;
      uint64_t v603 = (uint64_t) v37;
      TASSIGN(v602, v603);
      // pto: %t__tmp_v450
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v604 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v68);
      // pto: %t__tmp_v450
      ;
      uint64_t v605 = (uint64_t) v44;
      TASSIGN(v604, v605);
      // pto: %142
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v606;
      TRESHAPE(v606, v602);
      // pto: %143
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v607;
      TRESHAPE(v607, v604);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v607, v76, v606);
      // pto: %post_ids_float_inline1796__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v608 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v68);
      // pto: %post_ids_float_inline1796__ssa_v0
      ;
      uint64_t v609 = (uint64_t) v37;
      TASSIGN(v608, v609);
      pipe_barrier(PIPE_V);
      RoundMode v610 = RoundMode::CAST_ROUND;
      SaturationMode v611 = SaturationMode::ON;
      TCVT(v608, v604, v610, v611);
      // pto: %t__tmp_v451
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v612 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v68);
      // pto: %t__tmp_v451
      ;
      uint64_t v613 = (uint64_t) v59;
      TASSIGN(v612, v613);
      pipe_barrier(PIPE_V);
      TMULS(v612, v608, v94);
      // pto: %t__tmp_v452
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v614 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v68);
      // pto: %t__tmp_v452
      ;
      uint64_t v615 = (uint64_t) v59;
      TASSIGN(v614, v615);
      pipe_barrier(PIPE_V);
      RoundMode v616 = RoundMode::CAST_FLOOR;
      SaturationMode v617 = SaturationMode::ON;
      TCVT(v614, v612, v616, v617);
      // pto: %post_rows_inline1907__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v618 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v68);
      // pto: %post_rows_inline1907__ssa_v0
      ;
      uint64_t v619 = (uint64_t) v59;
      TASSIGN(v618, v619);
      pipe_barrier(PIPE_V);
      RoundMode v620 = RoundMode::CAST_ROUND;
      SaturationMode v621 = SaturationMode::ON;
      TCVT(v618, v614, v620, v621);
      // pto: %t__tmp_v453
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v622 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v68);
      // pto: %t__tmp_v453
      ;
      uint64_t v623 = (uint64_t) v59;
      TASSIGN(v622, v623);
      pipe_barrier(PIPE_V);
      TMULS(v622, v618, v96);
      // pto: %t__tmp_v454
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v624 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v68);
      // pto: %t__tmp_v454
      ;
      uint64_t v625 = (uint64_t) v37;
      TASSIGN(v624, v625);
      pipe_barrier(PIPE_V);
      TADD(v624, v608, v622);
      // pto: %t__tmp_v455
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v626 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v68);
      // pto: %t__tmp_v455
      ;
      uint64_t v627 = (uint64_t) v37;
      TASSIGN(v626, v627);
      pipe_barrier(PIPE_V);
      TADDS(v626, v624, v96);
      // pto: %post_ids_inline1626__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v628 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v68);
      // pto: %post_ids_inline1626__ssa_v0
      ;
      uint64_t v629 = (uint64_t) v37;
      TASSIGN(v628, v629);
      pipe_barrier(PIPE_V);
      RoundMode v630 = RoundMode::CAST_ROUND;
      SaturationMode v631 = SaturationMode::ON;
      TCVT(v628, v626, v630, v631);
      // pto: %post_tmp_inline1887__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v632 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v68);
      // pto: %post_tmp_inline1887__ssa_v0
      ;
      uint64_t v633 = (uint64_t) v59;
      TASSIGN(v632, v633);
      // pto: %t__tmp_v456
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v634 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v77);
      // pto: %t__tmp_v456
      ;
      uint64_t v635 = (uint64_t) v58;
      TASSIGN(v634, v635);
      // pto: %post_values_inline1625__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v636 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v68);
      // pto: %post_values_inline1625__ssa_v0
      ;
      uint64_t v637 = (uint64_t) v57;
      TASSIGN(v636, v637);
      pipe_barrier(PIPE_V);
      TGATHER(v636, v634, v628, v632);
      // pto: %t__tmp_v457
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v638 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v68);
      // pto: %t__tmp_v457
      ;
      uint64_t v639 = (uint64_t) v37;
      TASSIGN(v638, v639);
      pipe_barrier(PIPE_V);
      TMULS(v638, v636, v97);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      // pto: %144
      ;
      int64_t v640 = (int64_t) ((uint64_t) v498 * (uint64_t) v81);
      v638.SetValidShape(v69, v640);
      // pto: %145
      ;
      bool v641 = v476 == v80;
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
      if (v641) {
        // pto: %147
        ;
        int64_t v642 = (int64_t) ((uint64_t) v496 * (uint64_t) v81);
        // pto: %148
        ;
        int64_t v643 = v642 < v80 ? v80 : v642;
        // pto: %post_flat_inline1729__ssa_v0_pview
        ;
        const int64_t v644 = 0;
        // pto: %post_flat_inline1729__ssa_v0_pview
        ;
        __gm__ float* v645 = PTOAS__GLOBAL_TENSOR_DATA(v167);
        // pto: %post_flat_inline1729__ssa_v0_pview
        ;
        const int64_t v646 = 1;
        // pto: %post_flat_inline1729__ssa_v0_pview
        ;
        const int64_t v647 = 1;
        // pto: %post_flat_inline1729__ssa_v0_pview
        ;
        const int64_t v648 = 1;
        // pto: %post_flat_inline1729__ssa_v0_pview
        ;
        int64_t v649 = v69 * v159;
        // pto: %post_flat_inline1729__ssa_v0_pview
        ;
        int64_t v650 = v648 * v649;
        // pto: %post_flat_inline1729__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, -1> v651 = pto::Shape<1, 1, 1, 1, -1>(v646, v647, v648, v69, v640);
        // pto: %post_flat_inline1729__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v652 = pto::Stride<-1, -1, -1, -1, -1>(v647 * v650, v650, v649, v159, v69);
        // pto: %post_flat_inline1729__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v653 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v645 + (v644 + v80 * v159 + v643 * v69), v651, v652);
        TSTORE(v653, v638);
      };
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      // pto: %149
      ;
      int64_t v654 = v498 / v82;
      // pto: %150
      ;
      int64_t v655 = (int64_t) ((uint64_t) v654 * (uint64_t) v82);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      for (int64_t v656 = v80; v656 < v655; v656 += v82) {
        // pto: %151
        ;
        int64_t v657 = (int64_t) ((uint64_t) v496 + (uint64_t) v656);
        // pto: %152
        ;
        int64_t v658 = (int64_t) ((uint64_t) v656 + (uint64_t) v69);
        // pto: %153
        ;
        int64_t v659 = (int64_t) ((uint64_t) v496 + (uint64_t) v658);
        // pto: %flat_offset_mul
        ;
        int64_t v660 = (int64_t) ((uint64_t) v656 * (uint64_t) v73);
        // pto: %t__tmp_v458
        ;
        float v661 = v600.GetValue(v660);
        // pto: %154
        ;
        float v662 = v661 + v86;
        // pto: %156, %t__tmp_v459
        ;
        float v663 = v600.GetValue((int64_t) ((uint64_t) v660 + (uint64_t) v69));
        // pto: %157
        ;
        float v664 = v663 + v86;
        // pto: %159, %t__tmp_v460
        ;
        float v665 = v600.GetValue((int64_t) ((uint64_t) v660 + (uint64_t) v82));
        // pto: %160
        ;
        float v666 = v665 + v86;
        // pto: %162, %t__tmp_v461
        ;
        float v667 = v600.GetValue((int64_t) ((uint64_t) v660 + (uint64_t) v72));
        // pto: %163
        ;
        float v668 = v667 + v86;
        // pto: %t__tmp_v462
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v669 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %t__tmp_v462
        ;
        uint64_t v670 = (uint64_t) v60;
        TASSIGN(v669, v670);
        // pto: %164
        ;
        int64_t v671 = v657 < v80 ? v80 : v657;
        // pto: %165
        ;
        __gm__ bfloat16_t* v672 = PTOAS__GLOBAL_TENSOR_DATA(v115);
        // pto: %165
        ;
        const int64_t v673 = 0;
        // pto: %165
        ;
        const int64_t v674 = 16384;
        // pto: %165
        ;
        pto::Shape<1, 1, 1, 1, 4096> v675 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %165
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v676 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %165
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v677 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v672 + (v673 + v671 * v674), v675, v676);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v669, v677);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %t__tmp_v463
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v678 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %t__tmp_v463
        ;
        uint64_t v679 = (uint64_t) v61;
        TASSIGN(v678, v679);
        // pto: %167
        ;
        __gm__ bfloat16_t* v680 = PTOAS__GLOBAL_TENSOR_DATA(v115);
        // pto: %167
        ;
        const int64_t v681 = 4096;
        // pto: %167
        ;
        const int64_t v682 = 16384;
        // pto: %167
        ;
        pto::Shape<1, 1, 1, 1, 4096> v683 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %167
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v684 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %167
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v685 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v680 + (v681 + v671 * v682), v683, v684);
        TLOAD(v678, v685);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %t__tmp_v464
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v686 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %t__tmp_v464
        ;
        uint64_t v687 = (uint64_t) v62;
        TASSIGN(v686, v687);
        // pto: %169
        ;
        __gm__ bfloat16_t* v688 = PTOAS__GLOBAL_TENSOR_DATA(v115);
        // pto: %169
        ;
        const int64_t v689 = 8192;
        // pto: %169
        ;
        const int64_t v690 = 16384;
        // pto: %169
        ;
        pto::Shape<1, 1, 1, 1, 4096> v691 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %169
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v692 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %169
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v693 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v688 + (v689 + v671 * v690), v691, v692);
        TLOAD(v686, v693);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %t__tmp_v465
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v694 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %t__tmp_v465
        ;
        uint64_t v695 = (uint64_t) v63;
        TASSIGN(v694, v695);
        // pto: %171
        ;
        __gm__ bfloat16_t* v696 = PTOAS__GLOBAL_TENSOR_DATA(v115);
        // pto: %171
        ;
        const int64_t v697 = 12288;
        // pto: %171
        ;
        const int64_t v698 = 16384;
        // pto: %171
        ;
        pto::Shape<1, 1, 1, 1, 4096> v699 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %171
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v700 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %171
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v701 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v696 + (v697 + v671 * v698), v699, v700);
        TLOAD(v694, v701);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %174
        ;
        int64_t v702 = (int64_t) ((uint64_t) v658 * (uint64_t) v73);
        // pto: %172
        ;
        float v703 = v600.GetValue(v702);
        // pto: %176
        ;
        float v704 = v703 + v86;
        // pto: %180, %177
        ;
        float v705 = v600.GetValue((int64_t) ((uint64_t) v702 + (uint64_t) v69));
        // pto: %181
        ;
        float v706 = v705 + v86;
        // pto: %185, %182
        ;
        float v707 = v600.GetValue((int64_t) ((uint64_t) v702 + (uint64_t) v82));
        // pto: %186
        ;
        float v708 = v707 + v86;
        // pto: %190, %187
        ;
        float v709 = v600.GetValue((int64_t) ((uint64_t) v702 + (uint64_t) v72));
        // pto: %191
        ;
        float v710 = v709 + v86;
        // pto: %14
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v711 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %14
        ;
        uint64_t v712 = (uint64_t) v35;
        TASSIGN(v711, v712);
        // pto: %192
        ;
        int64_t v713 = v659 < v80 ? v80 : v659;
        // pto: %193
        ;
        __gm__ bfloat16_t* v714 = PTOAS__GLOBAL_TENSOR_DATA(v115);
        // pto: %193
        ;
        const int64_t v715 = 0;
        // pto: %193
        ;
        const int64_t v716 = 16384;
        // pto: %193
        ;
        pto::Shape<1, 1, 1, 1, 4096> v717 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %193
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v718 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %193
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v719 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v714 + (v715 + v713 * v716), v717, v718);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        TLOAD(v711, v719);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        // pto: %15
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v720 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %15
        ;
        uint64_t v721 = (uint64_t) v64;
        TASSIGN(v720, v721);
        // pto: %195
        ;
        __gm__ bfloat16_t* v722 = PTOAS__GLOBAL_TENSOR_DATA(v115);
        // pto: %195
        ;
        const int64_t v723 = 4096;
        // pto: %195
        ;
        const int64_t v724 = 16384;
        // pto: %195
        ;
        pto::Shape<1, 1, 1, 1, 4096> v725 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %195
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v726 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %195
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v727 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v722 + (v723 + v713 * v724), v725, v726);
        TLOAD(v720, v727);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        // pto: %16
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v728 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %16
        ;
        uint64_t v729 = (uint64_t) v65;
        TASSIGN(v728, v729);
        // pto: %197
        ;
        __gm__ bfloat16_t* v730 = PTOAS__GLOBAL_TENSOR_DATA(v115);
        // pto: %197
        ;
        const int64_t v731 = 8192;
        // pto: %197
        ;
        const int64_t v732 = 16384;
        // pto: %197
        ;
        pto::Shape<1, 1, 1, 1, 4096> v733 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %197
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v734 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %197
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v735 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v730 + (v731 + v713 * v732), v733, v734);
        TLOAD(v728, v735);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        // pto: %17
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v736 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %17
        ;
        uint64_t v737 = (uint64_t) v66;
        TASSIGN(v736, v737);
        // pto: %199
        ;
        __gm__ bfloat16_t* v738 = PTOAS__GLOBAL_TENSOR_DATA(v115);
        // pto: %199
        ;
        const int64_t v739 = 12288;
        // pto: %199
        ;
        const int64_t v740 = 16384;
        // pto: %199
        ;
        pto::Shape<1, 1, 1, 1, 4096> v741 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %199
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v742 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %199
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v743 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v738 + (v739 + v713 * v740), v741, v742);
        TLOAD(v736, v743);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        // pto: %scalar_x0_inline1609__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v744 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %scalar_x0_inline1609__ssa_v0
        ;
        uint64_t v745 = (uint64_t) v37;
        TASSIGN(v744, v745);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        RoundMode v746 = RoundMode::CAST_ROUND;
        SaturationMode v747 = SaturationMode::ON;
        TCVT(v744, v669, v746, v747);
        // pto: %scalar_y0_inline1608__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v748 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %scalar_y0_inline1608__ssa_v0
        ;
        uint64_t v749 = (uint64_t) v37;
        TASSIGN(v748, v749);
        pipe_barrier(PIPE_V);
        TMULS(v748, v744, v662);
        // pto: %scalar_x1_inline1607__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v750 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %scalar_x1_inline1607__ssa_v0
        ;
        uint64_t v751 = (uint64_t) v60;
        TASSIGN(v750, v751);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        RoundMode v752 = RoundMode::CAST_ROUND;
        SaturationMode v753 = SaturationMode::ON;
        TCVT(v750, v678, v752, v753);
        // pto: %scalar_y1_inline1895__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v754 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %scalar_y1_inline1895__ssa_v0
        ;
        uint64_t v755 = (uint64_t) v60;
        TASSIGN(v754, v755);
        pipe_barrier(PIPE_V);
        TMULS(v754, v750, v664);
        // pto: %scalar_sum01_inline1606__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v756 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %scalar_sum01_inline1606__ssa_v0
        ;
        uint64_t v757 = (uint64_t) v37;
        TASSIGN(v756, v757);
        pipe_barrier(PIPE_V);
        TADD(v756, v748, v754);
        // pto: %scalar_x2_inline1624__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v758 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %scalar_x2_inline1624__ssa_v0
        ;
        uint64_t v759 = (uint64_t) v60;
        TASSIGN(v758, v759);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        RoundMode v760 = RoundMode::CAST_ROUND;
        SaturationMode v761 = SaturationMode::ON;
        TCVT(v758, v686, v760, v761);
        // pto: %scalar_y2_inline1687__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v762 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %scalar_y2_inline1687__ssa_v0
        ;
        uint64_t v763 = (uint64_t) v60;
        TASSIGN(v762, v763);
        pipe_barrier(PIPE_V);
        TMULS(v762, v758, v666);
        // pto: %scalar_x3_inline1605__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v764 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %scalar_x3_inline1605__ssa_v0
        ;
        uint64_t v765 = (uint64_t) v61;
        TASSIGN(v764, v765);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        RoundMode v766 = RoundMode::CAST_ROUND;
        SaturationMode v767 = SaturationMode::ON;
        TCVT(v764, v694, v766, v767);
        // pto: %scalar_y3_inline1602__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v768 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %scalar_y3_inline1602__ssa_v0
        ;
        uint64_t v769 = (uint64_t) v61;
        TASSIGN(v768, v769);
        pipe_barrier(PIPE_V);
        TMULS(v768, v764, v668);
        // pto: %t__tmp_v466
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v770 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %t__tmp_v466
        ;
        uint64_t v771 = (uint64_t) v60;
        TASSIGN(v770, v771);
        pipe_barrier(PIPE_V);
        TADD(v770, v762, v768);
        // pto: %scalar_result_inline1601__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v772 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %scalar_result_inline1601__ssa_v0
        ;
        uint64_t v773 = (uint64_t) v37;
        TASSIGN(v772, v773);
        pipe_barrier(PIPE_V);
        TADD(v772, v756, v770);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %scalar_bf16_inline1628__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v774 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %scalar_bf16_inline1628__ssa_v0
        ;
        uint64_t v775 = (uint64_t) v37;
        TASSIGN(v774, v775);
        pipe_barrier(PIPE_V);
        RoundMode v776 = RoundMode::CAST_RINT;
        SaturationMode v777 = SaturationMode::ON;
        TCVT(v774, v772, v776, v777);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        // pto: %mixed_view_inline1689__ssa_v0_pview
        ;
        __gm__ bfloat16_t* v778 = PTOAS__GLOBAL_TENSOR_DATA(v176);
        // pto: %mixed_view_inline1689__ssa_v0_pview
        ;
        const int64_t v779 = 0;
        // pto: %mixed_view_inline1689__ssa_v0_pview
        ;
        const int64_t v780 = 4096;
        // pto: %mixed_view_inline1689__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, 4096> v781 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %mixed_view_inline1689__ssa_v0_pview
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v782 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %mixed_view_inline1689__ssa_v0_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v783 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v778 + (v779 + v671 * v780), v781, v782);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v783, v774);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        // pto: %18
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v784 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %18
        ;
        uint64_t v785 = (uint64_t) v67;
        TASSIGN(v784, v785);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
        RoundMode v786 = RoundMode::CAST_ROUND;
        SaturationMode v787 = SaturationMode::ON;
        TCVT(v784, v711, v786, v787);
        // pto: %19
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v788 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %19
        ;
        uint64_t v789 = (uint64_t) v67;
        TASSIGN(v788, v789);
        pipe_barrier(PIPE_V);
        TMULS(v788, v784, v704);
        // pto: %20
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v790 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %20
        ;
        uint64_t v791 = (uint64_t) v35;
        TASSIGN(v790, v791);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        RoundMode v792 = RoundMode::CAST_ROUND;
        SaturationMode v793 = SaturationMode::ON;
        TCVT(v790, v720, v792, v793);
        // pto: %21
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v794 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %21
        ;
        uint64_t v795 = (uint64_t) v35;
        TASSIGN(v794, v795);
        pipe_barrier(PIPE_V);
        TMULS(v794, v790, v706);
        // pto: %22
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v796 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %22
        ;
        uint64_t v797 = (uint64_t) v67;
        TASSIGN(v796, v797);
        pipe_barrier(PIPE_V);
        TADD(v796, v788, v794);
        // pto: %23
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v798 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %23
        ;
        uint64_t v799 = (uint64_t) v35;
        TASSIGN(v798, v799);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        RoundMode v800 = RoundMode::CAST_ROUND;
        SaturationMode v801 = SaturationMode::ON;
        TCVT(v798, v728, v800, v801);
        // pto: %24
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v802 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %24
        ;
        uint64_t v803 = (uint64_t) v35;
        TASSIGN(v802, v803);
        pipe_barrier(PIPE_V);
        TMULS(v802, v798, v708);
        // pto: %25
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v804 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %25
        ;
        uint64_t v805 = (uint64_t) v64;
        TASSIGN(v804, v805);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        RoundMode v806 = RoundMode::CAST_ROUND;
        SaturationMode v807 = SaturationMode::ON;
        TCVT(v804, v736, v806, v807);
        // pto: %26
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v808 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %26
        ;
        uint64_t v809 = (uint64_t) v64;
        TASSIGN(v808, v809);
        pipe_barrier(PIPE_V);
        TMULS(v808, v804, v710);
        // pto: %27
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v810 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %27
        ;
        uint64_t v811 = (uint64_t) v35;
        TASSIGN(v810, v811);
        pipe_barrier(PIPE_V);
        TADD(v810, v802, v808);
        // pto: %28
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v812 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %28
        ;
        uint64_t v813 = (uint64_t) v67;
        TASSIGN(v812, v813);
        pipe_barrier(PIPE_V);
        TADD(v812, v796, v810);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        // pto: %29
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v814 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %29
        ;
        uint64_t v815 = (uint64_t) v67;
        TASSIGN(v814, v815);
        pipe_barrier(PIPE_V);
        RoundMode v816 = RoundMode::CAST_RINT;
        SaturationMode v817 = SaturationMode::ON;
        TCVT(v814, v812, v816, v817);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %203
        ;
        __gm__ bfloat16_t* v818 = PTOAS__GLOBAL_TENSOR_DATA(v176);
        // pto: %203
        ;
        const int64_t v819 = 0;
        // pto: %203
        ;
        const int64_t v820 = 4096;
        // pto: %203
        ;
        pto::Shape<1, 1, 1, 1, 4096> v821 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %203
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v822 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %203
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v823 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v818 + (v819 + v713 * v820), v821, v822);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v823, v814);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
      };
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_S, PIPE_V, EVENT_ID1);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      // pto: %204, %205
      ;
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      if ((int64_t) ((uint64_t) v498 - (uint64_t) v655) == v69) {
        // pto: %206
        ;
        int64_t v824 = (int64_t) ((uint64_t) v496 + (uint64_t) v655);
        int64_t v825 = (int64_t) ((uint64_t) v654 * (uint64_t) v74);
        // pto: %207
        ;
        float v826 = v600.GetValue(v825);
        // pto: %210
        ;
        float v827 = v826 + v86;
        // pto: %213, %211
        ;
        float v828 = v600.GetValue((int64_t) ((uint64_t) v825 + (uint64_t) v69));
        // pto: %214
        ;
        float v829 = v828 + v86;
        // pto: %217, %215
        ;
        float v830 = v600.GetValue((int64_t) ((uint64_t) v825 + (uint64_t) v82));
        // pto: %218
        ;
        float v831 = v830 + v86;
        // pto: %221, %219
        ;
        float v832 = v600.GetValue((int64_t) ((uint64_t) v825 + (uint64_t) v72));
        // pto: %222
        ;
        float v833 = v832 + v86;
        // pto: %30
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v834 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %30
        ;
        uint64_t v835 = (uint64_t) v60;
        TASSIGN(v834, v835);
        // pto: %223
        ;
        int64_t v836 = v824 < v80 ? v80 : v824;
        // pto: %224
        ;
        __gm__ bfloat16_t* v837 = PTOAS__GLOBAL_TENSOR_DATA(v115);
        // pto: %224
        ;
        const int64_t v838 = 0;
        // pto: %224
        ;
        const int64_t v839 = 16384;
        // pto: %224
        ;
        pto::Shape<1, 1, 1, 1, 4096> v840 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %224
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v841 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %224
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v842 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v837 + (v838 + v836 * v839), v840, v841);
        TLOAD(v834, v842);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %31
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v843 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %31
        ;
        uint64_t v844 = (uint64_t) v37;
        TASSIGN(v843, v844);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v845 = RoundMode::CAST_ROUND;
        SaturationMode v846 = SaturationMode::ON;
        TCVT(v843, v834, v845, v846);
        // pto: %32
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v847 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %32
        ;
        uint64_t v848 = (uint64_t) v37;
        TASSIGN(v847, v848);
        pipe_barrier(PIPE_V);
        TMULS(v847, v843, v827);
        // pto: %33
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v849 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %33
        ;
        uint64_t v850 = (uint64_t) v61;
        TASSIGN(v849, v850);
        // pto: %226
        ;
        __gm__ bfloat16_t* v851 = PTOAS__GLOBAL_TENSOR_DATA(v115);
        // pto: %226
        ;
        const int64_t v852 = 4096;
        // pto: %226
        ;
        const int64_t v853 = 16384;
        // pto: %226
        ;
        pto::Shape<1, 1, 1, 1, 4096> v854 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %226
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v855 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %226
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v856 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v851 + (v852 + v836 * v853), v854, v855);
        TLOAD(v849, v856);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %34
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v857 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %34
        ;
        uint64_t v858 = (uint64_t) v60;
        TASSIGN(v857, v858);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v859 = RoundMode::CAST_ROUND;
        SaturationMode v860 = SaturationMode::ON;
        TCVT(v857, v849, v859, v860);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %35
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v861 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %35
        ;
        uint64_t v862 = (uint64_t) v60;
        TASSIGN(v861, v862);
        pipe_barrier(PIPE_V);
        TMULS(v861, v857, v829);
        // pto: %36
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v863 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %36
        ;
        uint64_t v864 = (uint64_t) v37;
        TASSIGN(v863, v864);
        pipe_barrier(PIPE_V);
        TADD(v863, v847, v861);
        // pto: %37
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v865 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %37
        ;
        uint64_t v866 = (uint64_t) v61;
        TASSIGN(v865, v866);
        // pto: %228
        ;
        __gm__ bfloat16_t* v867 = PTOAS__GLOBAL_TENSOR_DATA(v115);
        // pto: %228
        ;
        const int64_t v868 = 8192;
        // pto: %228
        ;
        const int64_t v869 = 16384;
        // pto: %228
        ;
        pto::Shape<1, 1, 1, 1, 4096> v870 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %228
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v871 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %228
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v872 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v867 + (v868 + v836 * v869), v870, v871);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v865, v872);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %38
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v873 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %38
        ;
        uint64_t v874 = (uint64_t) v60;
        TASSIGN(v873, v874);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v875 = RoundMode::CAST_ROUND;
        SaturationMode v876 = SaturationMode::ON;
        TCVT(v873, v865, v875, v876);
        // pto: %39
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v877 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %39
        ;
        uint64_t v878 = (uint64_t) v60;
        TASSIGN(v877, v878);
        pipe_barrier(PIPE_V);
        TMULS(v877, v873, v831);
        // pto: %40
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v879 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %40
        ;
        uint64_t v880 = (uint64_t) v67;
        TASSIGN(v879, v880);
        // pto: %230
        ;
        __gm__ bfloat16_t* v881 = PTOAS__GLOBAL_TENSOR_DATA(v115);
        // pto: %230
        ;
        const int64_t v882 = 12288;
        // pto: %230
        ;
        const int64_t v883 = 16384;
        // pto: %230
        ;
        pto::Shape<1, 1, 1, 1, 4096> v884 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %230
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v885 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %230
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v886 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v881 + (v882 + v836 * v883), v884, v885);
        TLOAD(v879, v886);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %41
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v887 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %41
        ;
        uint64_t v888 = (uint64_t) v61;
        TASSIGN(v887, v888);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v889 = RoundMode::CAST_ROUND;
        SaturationMode v890 = SaturationMode::ON;
        TCVT(v887, v879, v889, v890);
        // pto: %42
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v891 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %42
        ;
        uint64_t v892 = (uint64_t) v61;
        TASSIGN(v891, v892);
        pipe_barrier(PIPE_V);
        TMULS(v891, v887, v833);
        // pto: %43
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v893 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %43
        ;
        uint64_t v894 = (uint64_t) v60;
        TASSIGN(v893, v894);
        pipe_barrier(PIPE_V);
        TADD(v893, v877, v891);
        // pto: %44
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v895 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %44
        ;
        uint64_t v896 = (uint64_t) v37;
        TASSIGN(v895, v896);
        pipe_barrier(PIPE_V);
        TADD(v895, v863, v893);
        // pto: %45
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v897 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
        // pto: %45
        ;
        uint64_t v898 = (uint64_t) v37;
        TASSIGN(v897, v898);
        pipe_barrier(PIPE_V);
        RoundMode v899 = RoundMode::CAST_RINT;
        SaturationMode v900 = SaturationMode::ON;
        TCVT(v897, v895, v899, v900);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %233
        ;
        __gm__ bfloat16_t* v901 = PTOAS__GLOBAL_TENSOR_DATA(v176);
        // pto: %233
        ;
        const int64_t v902 = 0;
        // pto: %233
        ;
        const int64_t v903 = 4096;
        // pto: %233
        ;
        pto::Shape<1, 1, 1, 1, 4096> v904 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %233
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v905 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %233
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v906 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v901 + (v902 + v836 * v903), v904, v905);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v906, v897);
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