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

AICORE void hc_medium96_aic(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ bfloat16_t* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, __gm__ float* v10, __gm__ float* v11, __gm__ bfloat16_t* v12, __gm__ float* v13, int64_t v14, int32_t v15, int64_t v16, int64_t v17, int64_t v18, int64_t v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int64_t v28, int64_t v29, int64_t v30, int32_t v31, int32_t v32) {
  using T = float;

  #if defined(__DAV_CUBE__)
  const int64_t v33 = 768;
  const int32_t v34 = 0;
  // pto: %c0_i64
  const int64_t v35 = 0;
  // pto: %c262144_i64
  const int64_t v36 = 262144;
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
  // pto: %c8_index
  const int64_t v42 = 8;
  // pto: %c2048_index
  const int64_t v43 = 2048;
  // pto: %c0_index
  const int64_t v44 = 0;
  // pto: %c2_index
  const int64_t v45 = 2;
  // pto: %c1024_index
  const int64_t v46 = 1024;
  // pto: %c512_index
  const int64_t v47 = 512;
  // pto: %c256_index
  const int64_t v48 = 256;
  // pto: %cn256_index
  const int64_t v49 = -256;
  // pto: %40
  int64_t v50 = (int64_t) ((uint64_t) v29 * (uint64_t) v42);
  // pto: %mixes_partials_inline736__ssa_v0_view
  const int64_t v51 = 1;
  // pto: %mixes_partials_inline736__ssa_v0_view
  const int64_t v52 = 1;
  // pto: %mixes_partials_inline736__ssa_v0_view
  const int64_t v53 = 1;
  // pto: %mixes_partials_inline736__ssa_v0_view
  int64_t v54 = v50 * v38;
  // pto: %mixes_partials_inline736__ssa_v0_view
  int64_t v55 = v53 * v54;
  // pto: %mixes_partials_inline736__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v56 = pto::Shape<1, 1, 1, -1, -1>(v51, v52, v53, v50, v38);
  // pto: %mixes_partials_inline736__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v57 = pto::Stride<-1, -1, -1, -1, -1>(v52 * v55, v55, v54, v38, v39);
  // pto: %mixes_partials_inline736__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v58 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v56, v57);
  // pto: %hc_fn__ssa_v0_view
  const int64_t v59 = 1;
  // pto: %hc_fn__ssa_v0_view
  const int64_t v60 = 1;
  // pto: %hc_fn__ssa_v0_view
  const int64_t v61 = 1;
  // pto: %hc_fn__ssa_v0_view
  int64_t v62 = v40 * v41;
  // pto: %hc_fn__ssa_v0_view
  int64_t v63 = v61 * v62;
  // pto: %hc_fn__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v64 = pto::Shape<1, 1, 1, -1, -1>(v59, v60, v61, v40, v41);
  // pto: %hc_fn__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v65 = pto::Stride<-1, -1, -1, -1, -1>(v60 * v63, v63, v62, v41, v39);
  // pto: %hc_fn__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v66 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v64, v65);
  // pto: %mid_stream_product_inline933__phi_v10
  Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v67 = Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v38, v40);
  // pto: %mid_stream_product_inline933__phi_v10
  uint64_t v68 = (uint64_t) v35;
  TASSIGN(v67, v68);
  const int32_t v69 = 0;
  auto v70 = TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>(v13, v69, v34);
  // pto: %worker_inline761__ssa_v0
  // pto: %linear_workers_inline768__ssa_v0_idx
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  for (int64_t v71 = (int64_t) v31; v71 < v14; v71 += (int64_t) v15) {
    // pto: %12
    ;
    int64_t v72 = v71 % v42;
    // pto: %14, %15, %10, %11
    ;
    int64_t v73 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v72 * (uint64_t) v16) + (uint64_t) ((int64_t) (uint64_t) (v71 / v42) * (uint64_t) v38));
    // pto: %mid_stream_product_inline933__tile
    ;
    Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v74 = Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v38, v38);
    // pto: %mid_stream_product_inline933__tile
    ;
    uint64_t v75 = (uint64_t) v35;
    TASSIGN(v74, v75);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int64_t v76 = v44; v76 < v45; v76 += v39) {
      // pto: %13, %17, %16
      ;
      int64_t v77 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v72 * (uint64_t) v43) + (uint64_t) ((int64_t) (uint64_t) v76 * (uint64_t) v46));
      // pto: %mid_weight_chunk_inline735__tile
      ;
      Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v78 = Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v40, v46);
      // pto: %mid_weight_chunk_inline735__tile
      ;
      uint64_t v79 = (uint64_t) v36;
      TASSIGN(v78, v79);
      // pto: %18
      ;
      int64_t v80 = v77 < v44 ? v44 : v77;
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v81 = 0;
      // pto: %hc_fn__ssa_v0_pview
      ;
      __gm__ float* v82 = PTOAS__GLOBAL_TENSOR_DATA(v66);
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v83 = 1;
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v84 = 1;
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v85 = 1;
      // pto: %hc_fn__ssa_v0_pview
      ;
      int64_t v86 = v40 * v41;
      // pto: %hc_fn__ssa_v0_pview
      ;
      int64_t v87 = v85 * v86;
      // pto: %hc_fn__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 24, -1> v88 = pto::Shape<1, 1, 1, 24, -1>(v83, v84, v85, v40, v46);
      // pto: %hc_fn__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v89 = pto::Stride<-1, -1, -1, -1, -1>(v84 * v87, v87, v86, v41, v39);
      // pto: %hc_fn__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v90 = GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v82 + (v81 + v44 * v41 + v80 * v39), v88, v89);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v78, v90);
      // pto: %mid_gathered_chunk_inline792__tile
      ;
      const int32_t v91 = 32;
      // pto: %mid_gathered_chunk_inline792__tile
      ;
      const int32_t v92 = 1024;
      // pto: %mid_gathered_chunk_inline792__tile
      ;
      Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v93 = Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v91, v92);
      pipe_barrier(PIPE_MTE2);
      TPOP<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v70, v93);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %19
      ;
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      if (v76 == v44) {
        // pto: %mid_weight_chunk_inline735__tile_t
        ;
        Tile<TileType::Mat, float, 1024, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v94 = Tile<TileType::Mat, float, 1024, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v46, v40);
        // pto: %mid_weight_chunk_inline735__tile_t
        ;
        uint64_t v95 = (uint64_t) v36;
        TASSIGN(v94, v95);
        // pto: %mid_stream_product_inline933__tile_l0_init_storage
        ;
        Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v38, v38);
        // pto: %mid_stream_product_inline933__tile_l0_init_storage
        ;
        uint64_t v97 = (uint64_t) v35;
        TASSIGN(v96, v97);
        v96.SetValidShape(v38, v40);
        for (int64_t v98 = v44; v98 < v46; v98 += v47) {
          // pto: %mid_stream_product_inline933__tile_l0_a
          ;
          Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v99 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v38, v48);
          // pto: %mid_stream_product_inline933__tile_l0_a
          ;
          uint64_t v100 = (uint64_t) v35;
          TASSIGN(v99, v100);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          pipe_barrier(PIPE_MTE1);
          uint16_t v101 = static_cast<uint16_t>(v44);
          uint16_t v102 = static_cast<uint16_t>(v98);
          TEXTRACT(v99, v93, v101, v102);
          // pto: %20
          ;
          int64_t v103 = (int64_t) ((uint64_t) v46 - (uint64_t) v98);
          // pto: %21
          ;
          int64_t v104 = v103 < v48 ? v103 : v48;
          // pto: %mid_stream_product_inline933__tile_l0_b
          ;
          Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v105 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v104, v40);
          // pto: %mid_stream_product_inline933__tile_l0_b
          ;
          uint64_t v106 = (uint64_t) v35;
          TASSIGN(v105, v106);
          uint16_t v107 = static_cast<uint16_t>(v98);
          uint16_t v108 = static_cast<uint16_t>(v44);
          TEXTRACT(v105, v94, v107, v108);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          // pto: %0
          ;
          Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v109 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v38, v48);
          // pto: %0
          ;
          uint64_t v110 = (uint64_t) v37;
          TASSIGN(v109, v110);
          // pto: %22
          ;
          int64_t v111 = (int64_t) ((uint64_t) v98 + (uint64_t) v48);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
          uint16_t v112 = static_cast<uint16_t>(v44);
          uint16_t v113 = static_cast<uint16_t>(v111);
          TEXTRACT(v109, v93, v112, v113);
          int64_t v114 = (int64_t) ((uint64_t) v33 - (uint64_t) v98);
          // pto: %25
          ;
          int64_t v115 = v114 < v44 ? v44 : v114;
          // pto: %26
          ;
          int64_t v116 = v115 < v48 ? v115 : v48;
          // pto: %1
          ;
          Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v117 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v116, v40);
          // pto: %1
          ;
          uint64_t v118 = (uint64_t) v37;
          TASSIGN(v117, v118);
          uint16_t v119 = static_cast<uint16_t>(v111);
          uint16_t v120 = static_cast<uint16_t>(v44);
          TEXTRACT(v117, v94, v119, v120);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          // pto: %28
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          if (v98 == v44) {
            pipe_barrier(PIPE_M);
            TMATMUL(v96, v99, v105);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v96, v96, v99, v105);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          // pto: %29
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          if (v98 == v49) {
            pipe_barrier(PIPE_M);
            TMATMUL(v96, v109, v117);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v96, v96, v109, v117);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        };
      } else {
        // pto: %3
        ;
        Tile<TileType::Mat, float, 1024, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v121 = Tile<TileType::Mat, float, 1024, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v46, v40);
        // pto: %3
        ;
        uint64_t v122 = (uint64_t) v36;
        TASSIGN(v121, v122);
        for (int64_t v123 = v44; v123 < v46; v123 += v47) {
          // pto: %4
          ;
          Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v124 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v38, v48);
          // pto: %4
          ;
          uint64_t v125 = (uint64_t) v35;
          TASSIGN(v124, v125);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
          uint16_t v126 = static_cast<uint16_t>(v44);
          uint16_t v127 = static_cast<uint16_t>(v123);
          TEXTRACT(v124, v93, v126, v127);
          // pto: %31
          ;
          int64_t v128 = (int64_t) ((uint64_t) v46 - (uint64_t) v123);
          // pto: %32
          ;
          int64_t v129 = v128 < v48 ? v128 : v48;
          // pto: %5
          ;
          Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v130 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v129, v40);
          // pto: %5
          ;
          uint64_t v131 = (uint64_t) v35;
          TASSIGN(v130, v131);
          uint16_t v132 = static_cast<uint16_t>(v123);
          uint16_t v133 = static_cast<uint16_t>(v44);
          TEXTRACT(v130, v121, v132, v133);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
          // pto: %6
          ;
          Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v134 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v38, v48);
          // pto: %6
          ;
          uint64_t v135 = (uint64_t) v37;
          TASSIGN(v134, v135);
          // pto: %33
          ;
          int64_t v136 = (int64_t) ((uint64_t) v123 + (uint64_t) v48);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
          uint16_t v137 = static_cast<uint16_t>(v44);
          uint16_t v138 = static_cast<uint16_t>(v136);
          TEXTRACT(v134, v93, v137, v138);
          int64_t v139 = (int64_t) ((uint64_t) v33 - (uint64_t) v123);
          // pto: %36
          ;
          int64_t v140 = v139 < v44 ? v44 : v139;
          // pto: %37
          ;
          int64_t v141 = v140 < v48 ? v140 : v48;
          // pto: %7
          ;
          Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v142 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v141, v40);
          // pto: %7
          ;
          uint64_t v143 = (uint64_t) v37;
          TASSIGN(v142, v143);
          uint16_t v144 = static_cast<uint16_t>(v136);
          uint16_t v145 = static_cast<uint16_t>(v44);
          TEXTRACT(v142, v121, v144, v145);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
          pipe_barrier(PIPE_M);
          TMATMUL_ACC(v74, v74, v124, v130);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
          pipe_barrier(PIPE_M);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
          TMATMUL_ACC(v74, v74, v134, v142);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        };
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TFREE<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, TileSplitAxis::TILE_UP_DOWN>(v70);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    // pto: %39
    ;
    int64_t v146 = v73 < v44 ? v44 : v73;
    // pto: %mixes_partials_inline736__iter_v1_pview
    ;
    __gm__ float* v147 = PTOAS__GLOBAL_TENSOR_DATA(v58);
    // pto: %mixes_partials_inline736__iter_v1_pview
    ;
    const int64_t v148 = 0;
    // pto: %mixes_partials_inline736__iter_v1_pview
    ;
    const int64_t v149 = 32;
    // pto: %mixes_partials_inline736__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 32, 32> v150 = pto::Shape<1, 1, 1, 32, 32>();
    // pto: %mixes_partials_inline736__iter_v1_pview
    ;
    pto::Stride<1024, 1024, 1024, 32, 1> v151 = pto::Stride<1024, 1024, 1024, 32, 1>();
    // pto: %mixes_partials_inline736__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v152 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v147 + (v148 + v146 * v149), v150, v151);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v152, v74);
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

AICORE void hc_medium96_aiv(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ bfloat16_t* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, __gm__ float* v10, __gm__ float* v11, __gm__ bfloat16_t* v12, __gm__ float* v13, int64_t v14, int32_t v15, int64_t v16, int64_t v17, int64_t v18, int64_t v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int64_t v28, int64_t v29, int64_t v30, int32_t v31, int32_t v32, int32_t v33) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v34 = 0;
  // pto: %c65600_i64
  const int64_t v35 = 65600;
  // pto: %c64_i64
  const int64_t v36 = 64;
  // pto: %c131136_i64
  const int64_t v37 = 131136;
  // pto: %c1088_i64
  const int64_t v38 = 1088;
  // pto: %c1344_i64
  const int64_t v39 = 1344;
  // pto: %c1152_i64
  const int64_t v40 = 1152;
  const int64_t v41 = 96;
  // pto: %c576_i64
  const int64_t v42 = 576;
  // pto: %c832_i64
  const int64_t v43 = 832;
  // pto: %c320_i64
  const int64_t v44 = 320;
  // pto: %c1600_i64
  const int64_t v45 = 1600;
  // pto: %c1856_i64
  const int64_t v46 = 1856;
  // pto: %c2112_i64
  const int64_t v47 = 2112;
  // pto: %c2368_i64
  const int64_t v48 = 2368;
  // pto: %c1888_i64
  const int64_t v49 = 1888;
  // pto: %c1920_i64
  const int64_t v50 = 1920;
  // pto: %c1952_i64
  const int64_t v51 = 1952;
  // pto: %c1120_i64
  const int64_t v52 = 1120;
  // pto: %c114752_i64
  const int64_t v53 = 114752;
  // pto: %c192_i64
  const int64_t v54 = 192;
  // pto: %c16448_i64
  const int64_t v55 = 16448;
  // pto: %c32832_i64
  const int64_t v56 = 32832;
  // pto: %c41024_i64
  const int64_t v57 = 41024;
  // pto: %c98368_i64
  const int64_t v58 = 98368;
  // pto: %c81984_i64
  const int64_t v59 = 81984;
  // pto: %c90176_i64
  const int64_t v60 = 90176;
  // pto: %c106560_i64
  const int64_t v61 = 106560;
  // pto: %c49216_i64
  const int64_t v62 = 49216;
  // pto: %c32_index
  const int64_t v63 = 32;
  // pto: %c1_index
  const int64_t v64 = 1;
  // pto: %c24_index
  const int64_t v65 = 24;
  // pto: %c16384_index
  const int64_t v66 = 16384;
  // pto: %c3_index
  const int64_t v67 = 3;
  // pto: %c8_index
  const int64_t v68 = 8;
  const int64_t v69 = 16;
  // pto: %c4096_index
  const int64_t v70 = 4096;
  // pto: %c0_i32
  const int32_t v71 = 0;
  // pto: %c2048_index
  const int64_t v72 = 2048;
  // pto: %cst_39
  const float v73 = 0.0f;
  // pto: %c0_index
  const int64_t v74 = 0;
  // pto: %c2_index
  const int64_t v75 = 2;
  // pto: %c1024_index
  const int64_t v76 = 1024;
  // pto: %c4_index
  const int64_t v77 = 4;
  // pto: %c256_index
  const int64_t v78 = 256;
  // pto: %cst_45
  const float v79 = 6.10351563E-5f;
  // pto: %cst_46
  const float v80 = 9.99999997E-7f;
  // pto: %c64_index
  const int64_t v81 = 64;
  // pto: %c192_index
  const int64_t v82 = 192;
  // pto: %c128_index
  const int64_t v83 = 128;
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
  // pto: %212
  int64_t v93 = (int64_t) ((uint64_t) v29 * (uint64_t) v68);
  // pto: %rms_partials_inline705__ssa_v0_view
  const int64_t v94 = 1;
  // pto: %rms_partials_inline705__ssa_v0_view
  const int64_t v95 = 1;
  // pto: %rms_partials_inline705__ssa_v0_view
  const int64_t v96 = 1;
  // pto: %rms_partials_inline705__ssa_v0_view
  int64_t v97 = v93 * v64;
  // pto: %rms_partials_inline705__ssa_v0_view
  int64_t v98 = v96 * v97;
  // pto: %rms_partials_inline705__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v99 = pto::Shape<1, 1, 1, -1, -1>(v94, v95, v96, v93, v64);
  // pto: %rms_partials_inline705__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v100 = pto::Stride<-1, -1, -1, -1, -1>(v95 * v98, v98, v97, v64, v93);
  // pto: %rms_partials_inline705__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v101 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v99, v100);
  // pto: %x_flat_inline872__ssa_v0_view
  const int64_t v102 = 1;
  // pto: %x_flat_inline872__ssa_v0_view
  const int64_t v103 = 1;
  // pto: %x_flat_inline872__ssa_v0_view
  const int64_t v104 = 1;
  // pto: %x_flat_inline872__ssa_v0_view
  int64_t v105 = (int64_t) v30;
  // pto: %x_flat_inline872__ssa_v0_view
  int64_t v106 = v105 * v66;
  // pto: %x_flat_inline872__ssa_v0_view
  int64_t v107 = v104 * v106;
  // pto: %x_flat_inline872__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v108 = pto::Shape<1, 1, 1, -1, -1>(v102, v103, v104, v105, v66);
  // pto: %x_flat_inline872__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v109 = pto::Stride<-1, -1, -1, -1, -1>(v103 * v107, v107, v106, v66, v64);
  // pto: %x_flat_inline872__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v110 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v108, v109);
  // pto: %213
  int64_t v111 = (int64_t) ((uint64_t) v29 * (uint64_t) v63);
  // pto: %parts_2d_inline786__ssa_v0_view
  const int64_t v112 = 1;
  // pto: %parts_2d_inline786__ssa_v0_view
  const int64_t v113 = 1;
  // pto: %parts_2d_inline786__ssa_v0_view
  const int64_t v114 = 1;
  // pto: %parts_2d_inline786__ssa_v0_view
  int64_t v115 = v68 * v111;
  // pto: %parts_2d_inline786__ssa_v0_view
  int64_t v116 = v114 * v115;
  // pto: %parts_2d_inline786__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v117 = pto::Shape<1, 1, 1, -1, -1>(v112, v113, v114, v68, v111);
  // pto: %parts_2d_inline786__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v118 = pto::Stride<-1, -1, -1, -1, -1>(v113 * v116, v116, v115, v111, v64);
  // pto: %parts_2d_inline786__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v119 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v117, v118);
  // pto: %rms_2d_inline747__ssa_v0_view
  const int64_t v120 = 1;
  // pto: %rms_2d_inline747__ssa_v0_view
  const int64_t v121 = 1;
  // pto: %rms_2d_inline747__ssa_v0_view
  const int64_t v122 = 1;
  // pto: %rms_2d_inline747__ssa_v0_view
  int64_t v123 = (int64_t) v29;
  // pto: %rms_2d_inline747__ssa_v0_view
  int64_t v124 = v68 * v123;
  // pto: %rms_2d_inline747__ssa_v0_view
  int64_t v125 = v122 * v124;
  // pto: %rms_2d_inline747__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v126 = pto::Shape<1, 1, 1, -1, -1>(v120, v121, v122, v68, (int64_t) v29);
  // pto: %rms_2d_inline747__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v127 = pto::Stride<-1, -1, -1, -1, -1>(v121 * v125, v125, v124, v123, v64);
  // pto: %rms_2d_inline747__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v128 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v126, v127);
  // pto: %hc_base_2d_inline863__ssa_v0_view
  const int64_t v129 = 1;
  // pto: %hc_base_2d_inline863__ssa_v0_view
  const int64_t v130 = 1;
  // pto: %hc_base_2d_inline863__ssa_v0_view
  const int64_t v131 = 1;
  // pto: %hc_base_2d_inline863__ssa_v0_view
  int64_t v132 = v64 * v65;
  // pto: %hc_base_2d_inline863__ssa_v0_view
  int64_t v133 = v131 * v132;
  // pto: %hc_base_2d_inline863__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v134 = pto::Shape<1, 1, 1, -1, -1>(v129, v130, v131, v64, v65);
  // pto: %hc_base_2d_inline863__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v135 = pto::Stride<-1, -1, -1, -1, -1>(v130 * v133, v133, v132, v65, v64);
  // pto: %hc_base_2d_inline863__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v136 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v8, v134, v135);
  // pto: %comb_inline765__ssa_v0_view
  const int64_t v137 = 1;
  // pto: %comb_inline765__ssa_v0_view
  const int64_t v138 = 1;
  // pto: %comb_inline765__ssa_v0_view
  const int64_t v139 = 1;
  // pto: %comb_inline765__ssa_v0_view
  int64_t v140 = (int64_t) v30;
  // pto: %comb_inline765__ssa_v0_view
  int64_t v141 = v140 * v69;
  // pto: %comb_inline765__ssa_v0_view
  int64_t v142 = v139 * v141;
  // pto: %comb_inline765__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v143 = pto::Shape<1, 1, 1, -1, -1>(v137, v138, v139, v140, v69);
  // pto: %comb_inline765__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v144 = pto::Stride<-1, -1, -1, -1, -1>(v138 * v142, v142, v141, v69, v64);
  // pto: %comb_inline765__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v145 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v9, v143, v144);
  // pto: %hc_scale_2d_inline775__ssa_v0_view
  const int64_t v146 = 1;
  // pto: %hc_scale_2d_inline775__ssa_v0_view
  const int64_t v147 = 1;
  // pto: %hc_scale_2d_inline775__ssa_v0_view
  const int64_t v148 = 1;
  // pto: %hc_scale_2d_inline775__ssa_v0_view
  int64_t v149 = v64 * v67;
  // pto: %hc_scale_2d_inline775__ssa_v0_view
  int64_t v150 = v148 * v149;
  // pto: %hc_scale_2d_inline775__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v151 = pto::Shape<1, 1, 1, -1, -1>(v146, v147, v148, v64, v67);
  // pto: %hc_scale_2d_inline775__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v152 = pto::Stride<-1, -1, -1, -1, -1>(v147 * v150, v150, v149, v67, v64);
  // pto: %hc_scale_2d_inline775__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v153 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v10, v151, v152);
  // pto: %214
  int64_t v154 = (int64_t) ((uint64_t) v30 * (uint64_t) v77);
  // pto: %post_flat_inline909__ssa_v0_view
  const int64_t v155 = 1;
  // pto: %post_flat_inline909__ssa_v0_view
  const int64_t v156 = 1;
  // pto: %post_flat_inline909__ssa_v0_view
  const int64_t v157 = 1;
  // pto: %post_flat_inline909__ssa_v0_view
  int64_t v158 = v64 * v154;
  // pto: %post_flat_inline909__ssa_v0_view
  int64_t v159 = v157 * v158;
  // pto: %post_flat_inline909__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v160 = pto::Shape<1, 1, 1, -1, -1>(v155, v156, v157, v64, v154);
  // pto: %post_flat_inline909__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v161 = pto::Stride<-1, -1, -1, -1, -1>(v156 * v159, v159, v158, v154, v64);
  // pto: %post_flat_inline909__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v162 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v11, v160, v161);
  // pto: %mixed_view_inline708__ssa_v0_view
  const int64_t v163 = 1;
  // pto: %mixed_view_inline708__ssa_v0_view
  const int64_t v164 = 1;
  // pto: %mixed_view_inline708__ssa_v0_view
  const int64_t v165 = 1;
  // pto: %mixed_view_inline708__ssa_v0_view
  int64_t v166 = (int64_t) v30;
  // pto: %mixed_view_inline708__ssa_v0_view
  int64_t v167 = v166 * v70;
  // pto: %mixed_view_inline708__ssa_v0_view
  int64_t v168 = v165 * v167;
  // pto: %mixed_view_inline708__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v169 = pto::Shape<1, 1, 1, -1, -1>(v163, v164, v165, v166, v70);
  // pto: %mixed_view_inline708__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v170 = pto::Stride<-1, -1, -1, -1, -1>(v164 * v168, v168, v167, v70, v64);
  // pto: %mixed_view_inline708__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v171 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v12, v169, v170);
  const int32_t v172 = 0;
  auto v173 = TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>(v13, v172, v71);
  // pto: %worker_inline761__ssa_v0
  int64_t v174 = (int64_t) v31;
  // pto: %linear_workers_inline768__ssa_v0_idx
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  set_flag(PIPE_S, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
  for (int64_t v175 = v174; v175 < v14; v175 += (int64_t) v15) {
    // pto: %36, %37
    ;
    int64_t v176 = (int64_t) ((uint64_t) (v175 / v68) * (uint64_t) v63);
    // pto: %38
    ;
    int64_t v177 = v175 % v68;
    // pto: %mid_stream_rms_total_inline845__tile
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v178 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v69);
    // pto: %mid_stream_rms_total_inline845__tile
    ;
    uint64_t v179 = (uint64_t) v34;
    TASSIGN(v178, v179);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TEXPANDS(v178, v73);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    for (int64_t v180 = v74; v180 < v75; v180 += v64) {
      // pto: %39, %43, %42
      ;
      int64_t v181 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v177 * (uint64_t) v72) + (uint64_t) ((int64_t) (uint64_t) v180 * (uint64_t) v76));
      // pto: %45, %lane_inline817__ssa_v0, %44
      ;
      int64_t v182 = (int64_t) ((uint64_t) v176 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) v33) * (uint64_t) v69));
      // pto: %46
      ;
      int64_t v183 = (int64_t) ((uint64_t) v17 - (uint64_t) v182);
      // pto: %47
      ;
      int64_t v184 = v183 < v69 ? v183 : v69;
      // pto: %48
      ;
      int64_t v185 = v184 < v74 ? v74 : v184;
      // pto: %mid_stream_source_inline698__tile
      ;
      Tile<TileType::Vec, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v186 = Tile<TileType::Vec, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v185, v76);
      // pto: %mid_stream_source_inline698__tile
      ;
      uint64_t v187 = (uint64_t) v35;
      TASSIGN(v186, v187);
      // pto: %49
      ;
      int64_t v188 = v182 < v74 ? v74 : v182;
      // pto: %50
      ;
      int64_t v189 = v181 < v74 ? v74 : v181;
      // pto: %x_flat_inline872__ssa_v0_pview
      ;
      const int64_t v190 = 0;
      // pto: %x_flat_inline872__ssa_v0_pview
      ;
      __gm__ bfloat16_t* v191 = PTOAS__GLOBAL_TENSOR_DATA(v110);
      // pto: %x_flat_inline872__ssa_v0_pview
      ;
      const int64_t v192 = 1;
      // pto: %x_flat_inline872__ssa_v0_pview
      ;
      const int64_t v193 = 1;
      // pto: %x_flat_inline872__ssa_v0_pview
      ;
      const int64_t v194 = 1;
      // pto: %x_flat_inline872__ssa_v0_pview
      ;
      int64_t v195 = v185 * v66;
      // pto: %x_flat_inline872__ssa_v0_pview
      ;
      int64_t v196 = v194 * v195;
      // pto: %x_flat_inline872__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, -1> v197 = pto::Shape<1, 1, 1, -1, -1>(v192, v193, v194, v185, v76);
      // pto: %x_flat_inline872__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v198 = pto::Stride<-1, -1, -1, -1, -1>(v193 * v196, v196, v195, v66, v64);
      // pto: %x_flat_inline872__ssa_v0_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v199 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v191 + (v190 + v188 * v66 + v189 * v64), v197, v198);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v186, v199);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %mid_stream_widened_inline899__tile
      ;
      Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v200 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v185, v76);
      // pto: %mid_stream_widened_inline899__tile
      ;
      uint64_t v201 = (uint64_t) v36;
      TASSIGN(v200, v201);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      RoundMode v202 = RoundMode::CAST_ROUND;
      SaturationMode v203 = SaturationMode::ON;
      TCVT(v200, v186, v202, v203);
      v200.SetValidShape(v185, v76);
      // pto: %mid_stream_clean_inline866__tile
      ;
      Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v204 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v69, v76);
      // pto: %mid_stream_clean_inline866__tile
      ;
      uint64_t v205 = (uint64_t) v36;
      TASSIGN(v204, v205);
      pipe_barrier(PIPE_V);
      TFILLPAD<pto::TFillPadMode::InPlace>(v204, v200);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      v204.SetValidShape(v69, v76);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TPUSH<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v173, v204);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      // pto: %0
      ;
      Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v206 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v69, v76);
      // pto: %0
      ;
      uint64_t v207 = (uint64_t) v36;
      TASSIGN(v206, v207);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      TMUL(v206, v204, v204);
      // pto: %tmp_tile
      ;
      Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v208 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v76);
      // pto: %tmp_tile
      ;
      uint64_t v209 = (uint64_t) v35;
      TASSIGN(v208, v209);
      // pto: %1
      ;
      Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v210 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v64);
      // pto: %1
      ;
      uint64_t v211 = (uint64_t) v37;
      TASSIGN(v210, v211);
      pipe_barrier(PIPE_V);
      TROWSUM(v210, v206, v208);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      // pto: %mid_stream_sq_inline694__tile
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v212 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v69);
      // pto: %mid_stream_sq_inline694__tile
      ;
      uint64_t v213 = (uint64_t) v37;
      TASSIGN(v212, v213);
      // pto: %2
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v214 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v69);
      // pto: %2
      ;
      uint64_t v215 = (uint64_t) v34;
      TASSIGN(v214, v215);
      pipe_barrier(PIPE_V);
      TADD(v214, v178, v212);
    };
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %40, %41, %52, %rms_lane_inline893__ssa_v0, %51
    ;
    int64_t v216 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v177 * (uint64_t) v16) + (uint64_t) v176) + (uint64_t) ((int64_t) (uint64_t) ((int64_t) v33) * (uint64_t) v69));
    // pto: %3
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v217 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v64);
    // pto: %3
    ;
    uint64_t v218 = (uint64_t) v34;
    TASSIGN(v217, v218);
    // pto: %53
    ;
    int64_t v219 = v216 < v74 ? v74 : v216;
    // pto: %rms_partials_inline705__iter_v1_pview
    ;
    const int64_t v220 = 0;
    // pto: %rms_partials_inline705__iter_v1_pview
    ;
    __gm__ float* v221 = PTOAS__GLOBAL_TENSOR_DATA(v101);
    // pto: %rms_partials_inline705__iter_v1_pview
    ;
    const int64_t v222 = 1;
    // pto: %rms_partials_inline705__iter_v1_pview
    ;
    const int64_t v223 = 1;
    // pto: %rms_partials_inline705__iter_v1_pview
    ;
    const int64_t v224 = 1;
    // pto: %rms_partials_inline705__iter_v1_pview
    ;
    int64_t v225 = v69 * v64;
    // pto: %rms_partials_inline705__iter_v1_pview
    ;
    int64_t v226 = v224 * v225;
    // pto: %rms_partials_inline705__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 16, 1> v227 = pto::Shape<1, 1, 1, 16, 1>(v222, v223, v224, v69, v64);
    // pto: %rms_partials_inline705__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v228 = pto::Stride<-1, -1, -1, -1, -1>(v223 * v226, v226, v225, v64, v93);
    // pto: %rms_partials_inline705__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v229 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v221 + (v220 + v219 * v64 + v74 * v93), v227, v228);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    TSTORE(v229, v217);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  pipe_barrier(PIPE_ALL);
  dsb(DSB_DDR);
  SYNCALL<SyncCoreType::Mix>();
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  // pto: %54, %55, %tail_lane_inline743__ssa_v0
  int64_t v230 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v174 * (uint64_t) v75) + (uint64_t) ((int64_t) v33));
  // pto: %56
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  if (v230 < v18) {
    // pto: %58
    ;
    int64_t v231 = v230 < v20 ? v230 : v20;
    // pto: %57, %59
    ;
    int64_t v232 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v230 * (uint64_t) v19) + (uint64_t) v231);
    // pto: %61
    ;
    int64_t v233 = (int64_t) ((uint64_t) v20 - (uint64_t) v230);
    // pto: %62
    ;
    int64_t v234 = v233 < v74 ? v74 : v233;
    // pto: %63
    ;
    int64_t v235 = v234 < v64 ? v234 : v64;
    // pto: %60, %64
    ;
    int64_t v236 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v232 + (uint64_t) v19) + (uint64_t) v235);
    for (int64_t v237 = v232; v237 < v236; v237 += v77) {
      // pto: %scale2_inline718__tile
      ;
      float v238 = (v5)[v75];
      // pto: %65
      ;
      int64_t v239 = (int64_t) ((uint64_t) v236 - (uint64_t) v237);
      // pto: %66
      ;
      int64_t v240 = v239 < v77 ? v239 : v77;
      // pto: %67
      ;
      int64_t v241 = (int64_t) ((uint64_t) v240 * (uint64_t) v63);
      // pto: %c_parts_load_inline738__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v242 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v241);
      // pto: %c_parts_load_inline738__ssa_v0
      ;
      uint64_t v243 = (uint64_t) v36;
      TASSIGN(v242, v243);
      // pto: %68
      ;
      int64_t v244 = (int64_t) ((uint64_t) v237 * (uint64_t) v63);
      // pto: %69
      ;
      int64_t v245 = v244 < v74 ? v74 : v244;
      // pto: %parts_2d_inline786__ssa_v0_pview
      ;
      const int64_t v246 = 0;
      // pto: %parts_2d_inline786__ssa_v0_pview
      ;
      __gm__ float* v247 = PTOAS__GLOBAL_TENSOR_DATA(v119);
      // pto: %parts_2d_inline786__ssa_v0_pview
      ;
      const int64_t v248 = 1;
      // pto: %parts_2d_inline786__ssa_v0_pview
      ;
      const int64_t v249 = 1;
      // pto: %parts_2d_inline786__ssa_v0_pview
      ;
      const int64_t v250 = 1;
      // pto: %parts_2d_inline786__ssa_v0_pview
      ;
      int64_t v251 = v68 * v111;
      // pto: %parts_2d_inline786__ssa_v0_pview
      ;
      int64_t v252 = v250 * v251;
      // pto: %parts_2d_inline786__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, -1> v253 = pto::Shape<1, 1, 1, 8, -1>(v248, v249, v250, v68, v241);
      // pto: %parts_2d_inline786__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v254 = pto::Stride<-1, -1, -1, -1, -1>(v249 * v252, v252, v251, v111, v64);
      // pto: %parts_2d_inline786__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v255 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v247 + (v246 + v74 * v111 + v245 * v64), v253, v254);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      TLOAD(v242, v255);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %t__tmp_v166
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v256 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v68, v78);
      // pto: %t__tmp_v166
      ;
      uint64_t v257 = (uint64_t) v36;
      TASSIGN(v256, v257);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      TFILLPAD<pto::TFillPadMode::InPlace>(v256, v242);
      v256.SetValidShape(v68, v78);
      // pto: %t__tmp_v167
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v258 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v78);
      // pto: %t__tmp_v167
      ;
      uint64_t v259 = (uint64_t) v36;
      TASSIGN(v258, v259);
      pipe_barrier(PIPE_V);
      TCOLSUM(v258, v256);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
      // pto: %c_total_inline818__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v260 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v63);
      // pto: %c_total_inline818__ssa_v0
      ;
      uint64_t v261 = (uint64_t) v36;
      TASSIGN(v260, v261);
      // pto: %c_sq_load_inline757__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v262 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v240);
      // pto: %c_sq_load_inline757__ssa_v0
      ;
      uint64_t v263 = (uint64_t) v38;
      TASSIGN(v262, v263);
      // pto: %71
      ;
      int64_t v264 = v237 < v74 ? v74 : v237;
      // pto: %rms_2d_inline747__ssa_v0_pview
      ;
      int64_t v265 = (int64_t) v29;
      // pto: %rms_2d_inline747__ssa_v0_pview
      ;
      const int64_t v266 = 0;
      // pto: %rms_2d_inline747__ssa_v0_pview
      ;
      __gm__ float* v267 = PTOAS__GLOBAL_TENSOR_DATA(v128);
      // pto: %rms_2d_inline747__ssa_v0_pview
      ;
      const int64_t v268 = 1;
      // pto: %rms_2d_inline747__ssa_v0_pview
      ;
      const int64_t v269 = 1;
      // pto: %rms_2d_inline747__ssa_v0_pview
      ;
      const int64_t v270 = 1;
      // pto: %rms_2d_inline747__ssa_v0_pview
      ;
      int64_t v271 = v68 * v265;
      // pto: %rms_2d_inline747__ssa_v0_pview
      ;
      int64_t v272 = v270 * v271;
      // pto: %rms_2d_inline747__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, -1> v273 = pto::Shape<1, 1, 1, 8, -1>(v268, v269, v270, v68, v240);
      // pto: %rms_2d_inline747__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v274 = pto::Stride<-1, -1, -1, -1, -1>(v269 * v272, v272, v271, v265, v64);
      // pto: %rms_2d_inline747__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v275 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v267 + (v266 + v74 * v265 + v264 * v64), v273, v274);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
      TLOAD(v262, v275);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %t__tmp_v168
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v276 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v68, v68);
      // pto: %t__tmp_v168
      ;
      uint64_t v277 = (uint64_t) v38;
      TASSIGN(v276, v277);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      TFILLPAD<pto::TFillPadMode::InPlace>(v276, v262);
      v276.SetValidShape(v68, v68);
      // pto: %c_rms_tmp_inline691__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v278 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %c_rms_tmp_inline691__ssa_v0
      ;
      uint64_t v279 = (uint64_t) v39;
      TASSIGN(v278, v279);
      // pto: %t__tmp_v169
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v280 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v169
      ;
      uint64_t v281 = (uint64_t) v38;
      TASSIGN(v280, v281);
      pipe_barrier(PIPE_V);
      TCOLSUM(v280, v276);
      // pto: %t__tmp_v170
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v282 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v170
      ;
      uint64_t v283 = (uint64_t) v38;
      TASSIGN(v282, v283);
      pipe_barrier(PIPE_V);
      TMULS(v282, v280, v79);
      // pto: %t__tmp_v171
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v284 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v171
      ;
      uint64_t v285 = (uint64_t) v38;
      TASSIGN(v284, v285);
      pipe_barrier(PIPE_V);
      TADDS(v284, v282, v80);
      // pto: %t__tmp_v172
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v286 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v172
      ;
      uint64_t v287 = (uint64_t) v40;
      TASSIGN(v286, v287);
      pipe_barrier(PIPE_V);
      TRSQRT(v286, v284, v278);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      // pto: %c_inv_inline702__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v288 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v64);
      // pto: %c_inv_inline702__ssa_v0
      ;
      uint64_t v289 = (uint64_t) v40;
      TASSIGN(v288, v289);
      // pto: %comb_inv_inline803__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v290 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v64);
      // pto: %comb_inv_inline803__ssa_v0
      ;
      uint64_t v291 = (uint64_t) v40;
      TASSIGN(v290, v291);
      // pto: %t__tmp_v173
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v292 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v69);
      // pto: %t__tmp_v173
      ;
      uint64_t v293 = (uint64_t) v41;
      TASSIGN(v292, v293);
      // pto: %slice_view
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 16, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v294;
      // pto: %slice_view
      ;
      uint64_t v295 = (uint64_t) v41;
      TASSIGN(v294, v295);
      // pto: %raw_inline862__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v296 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v69);
      // pto: %raw_inline862__ssa_v0
      ;
      uint64_t v297 = (uint64_t) v36;
      TASSIGN(v296, v297);
      TADDS(v296, v294, v73);
      // pto: %base_inline905__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v298 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v69);
      // pto: %base_inline905__ssa_v0
      ;
      uint64_t v299 = (uint64_t) v38;
      TASSIGN(v298, v299);
      // pto: %hc_base_2d_inline863__ssa_v0_pview
      ;
      __gm__ float* v300 = PTOAS__GLOBAL_TENSOR_DATA(v136);
      // pto: %hc_base_2d_inline863__ssa_v0_pview
      ;
      unsigned v301 = 8;
      // pto: %hc_base_2d_inline863__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 16> v302 = pto::Shape<1, 1, 1, 1, 16>();
      // pto: %hc_base_2d_inline863__ssa_v0_pview
      ;
      pto::Stride<24, 24, 24, 24, 1> v303 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %hc_base_2d_inline863__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v304 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v300 + v301, v302, v303);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      TLOAD(v298, v304);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      // pto: %t__tmp_v174
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v305 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v69);
      // pto: %t__tmp_v174
      ;
      uint64_t v306 = (uint64_t) v42;
      TASSIGN(v305, v306);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v305, v296, v290);
      // pto: %t__tmp_v175
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v307 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v69);
      // pto: %t__tmp_v175
      ;
      uint64_t v308 = (uint64_t) v42;
      TASSIGN(v307, v308);
      pipe_barrier(PIPE_V);
      TMULS(v307, v305, v238);
      // pto: %t__tmp_v176
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v309 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v69);
      // pto: %t__tmp_v176
      ;
      uint64_t v310 = (uint64_t) v36;
      TASSIGN(v309, v310);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      TCOLEXPAND(v309, v298);
      // pto: %logits_inline690__ssa_v1
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v311 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v69);
      // pto: %logits_inline690__ssa_v1
      ;
      uint64_t v312 = (uint64_t) v43;
      TASSIGN(v311, v312);
      pipe_barrier(PIPE_V);
      TADD(v311, v307, v309);
      // pto: %t__tmp_v177
      ;
      Tile<TileType::Vec, float, 4, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v313 = Tile<TileType::Vec, float, 4, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v77, v69);
      // pto: %t__tmp_v177
      ;
      uint64_t v314 = (uint64_t) v43;
      TASSIGN(v313, v314);
      // pto: %regular_logits_inline880__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v315 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %regular_logits_inline880__ssa_v0
      ;
      uint64_t v316 = (uint64_t) v43;
      TASSIGN(v315, v316);
      // pto: %ids_inline824__ci_tmp_v0
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v317 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v82);
      // pto: %ids_inline824__ci_tmp_v0
      ;
      uint64_t v318 = (uint64_t) v36;
      TASSIGN(v317, v318);
      // pto: %ids_inline824__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v319 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %ids_inline824__ssa_v1
      ;
      uint64_t v320 = (uint64_t) v39;
      TASSIGN(v319, v320);
      // pto: %ci_tmp_view
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v321;
      TRESHAPE(v321, v317);
      // pto: %ci_dst_view
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v322;
      TRESHAPE(v322, v319);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v322, v71, v321);
      // pto: %tmp_inline825__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v323 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %tmp_inline825__ssa_v1
      ;
      uint64_t v324 = (uint64_t) v36;
      TASSIGN(v323, v324);
      // pto: %ids16_inline830__ssa_v1
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v325 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v83);
      // pto: %ids16_inline830__ssa_v1
      ;
      uint64_t v326 = (uint64_t) v39;
      TASSIGN(v325, v326);
      // pto: %xor_tmp16_inline924__ssa_v1
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v327 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v83);
      // pto: %xor_tmp16_inline924__ssa_v1
      ;
      uint64_t v328 = (uint64_t) v44;
      TASSIGN(v327, v328);
      // pto: %t__tmp_v178
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v329 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v178
      ;
      uint64_t v330 = (uint64_t) v42;
      TASSIGN(v329, v330);
      pipe_barrier(PIPE_V);
      TEXPANDS(v329, v84);
      // pto: %ri1_mask_inline840__ssa_v1
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v331 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v83);
      // pto: %ri1_mask_inline840__ssa_v1
      ;
      uint64_t v332 = (uint64_t) v42;
      TASSIGN(v331, v332);
      // pto: %t__tmp_v179
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v333 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v83);
      // pto: %t__tmp_v179
      ;
      uint64_t v334 = (uint64_t) v45;
      TASSIGN(v333, v334);
      pipe_barrier(PIPE_V);
      TXOR(v333, v325, v331, v327);
      // pto: %ri1_inline769__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v335 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %ri1_inline769__ssa_v1
      ;
      uint64_t v336 = (uint64_t) v45;
      TASSIGN(v335, v336);
      // pto: %t__tmp_v180
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v337 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v180
      ;
      uint64_t v338 = (uint64_t) v42;
      TASSIGN(v337, v338);
      pipe_barrier(PIPE_V);
      TEXPANDS(v337, v85);
      // pto: %ri2_mask_inline882__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v339 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v83);
      // pto: %ri2_mask_inline882__ssa_v0
      ;
      uint64_t v340 = (uint64_t) v42;
      TASSIGN(v339, v340);
      // pto: %t__tmp_v181
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v341 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v83);
      // pto: %t__tmp_v181
      ;
      uint64_t v342 = (uint64_t) v46;
      TASSIGN(v341, v342);
      pipe_barrier(PIPE_V);
      TXOR(v341, v325, v339, v327);
      // pto: %ri2_inline884__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v343 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %ri2_inline884__ssa_v0
      ;
      uint64_t v344 = (uint64_t) v46;
      TASSIGN(v343, v344);
      // pto: %t__tmp_v182
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v345 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v182
      ;
      uint64_t v346 = (uint64_t) v42;
      TASSIGN(v345, v346);
      pipe_barrier(PIPE_V);
      TEXPANDS(v345, v86);
      // pto: %ci1_mask_inline929__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v347 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v83);
      // pto: %ci1_mask_inline929__ssa_v0
      ;
      uint64_t v348 = (uint64_t) v42;
      TASSIGN(v347, v348);
      // pto: %t__tmp_v183
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v349 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v83);
      // pto: %t__tmp_v183
      ;
      uint64_t v350 = (uint64_t) v47;
      TASSIGN(v349, v350);
      pipe_barrier(PIPE_V);
      TXOR(v349, v325, v347, v327);
      // pto: %ci1_inline886__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v351 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %ci1_inline886__ssa_v0
      ;
      uint64_t v352 = (uint64_t) v47;
      TASSIGN(v351, v352);
      // pto: %t__tmp_v184
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v353 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v184
      ;
      uint64_t v354 = (uint64_t) v42;
      TASSIGN(v353, v354);
      pipe_barrier(PIPE_V);
      TEXPANDS(v353, v87);
      // pto: %ci2_mask_inline901__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v355 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v83);
      // pto: %ci2_mask_inline901__ssa_v0
      ;
      uint64_t v356 = (uint64_t) v42;
      TASSIGN(v355, v356);
      // pto: %t__tmp_v185
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v357 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v83);
      // pto: %t__tmp_v185
      ;
      uint64_t v358 = (uint64_t) v48;
      TASSIGN(v357, v358);
      pipe_barrier(PIPE_V);
      TXOR(v357, v325, v355, v327);
      // pto: %ci2_inline888__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v359 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %ci2_inline888__ssa_v0
      ;
      uint64_t v360 = (uint64_t) v48;
      TASSIGN(v359, v360);
      // pto: %t__tmp_v186
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v361 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v186
      ;
      uint64_t v362 = (uint64_t) v44;
      TASSIGN(v361, v362);
      pipe_barrier(PIPE_V);
      TGATHER(v361, v315, v335, v323);
      // pto: %max2_inline852__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v363 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %max2_inline852__ssa_v1
      ;
      uint64_t v364 = (uint64_t) v44;
      TASSIGN(v363, v364);
      pipe_barrier(PIPE_V);
      TMAX(v363, v315, v361);
      // pto: %t__tmp_v187
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v365 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v187
      ;
      uint64_t v366 = (uint64_t) v42;
      TASSIGN(v365, v366);
      pipe_barrier(PIPE_V);
      TGATHER(v365, v363, v343, v323);
      // pto: %max4_inline854__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v367 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %max4_inline854__ssa_v1
      ;
      uint64_t v368 = (uint64_t) v44;
      TASSIGN(v367, v368);
      pipe_barrier(PIPE_V);
      TMAX(v367, v363, v365);
      // pto: %t__tmp_v188
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v369 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v188
      ;
      uint64_t v370 = (uint64_t) v44;
      TASSIGN(v369, v370);
      pipe_barrier(PIPE_V);
      TSUB(v369, v315, v367);
      // pto: %exps_inline855__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v371 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %exps_inline855__ssa_v1
      ;
      uint64_t v372 = (uint64_t) v44;
      TASSIGN(v371, v372);
      pipe_barrier(PIPE_V);
      TEXP(v371, v369);
      // pto: %t__tmp_v189
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v373 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v189
      ;
      uint64_t v374 = (uint64_t) v42;
      TASSIGN(v373, v374);
      pipe_barrier(PIPE_V);
      TGATHER(v373, v371, v335, v323);
      // pto: %sum2_inline858__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v375 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %sum2_inline858__ssa_v1
      ;
      uint64_t v376 = (uint64_t) v42;
      TASSIGN(v375, v376);
      pipe_barrier(PIPE_V);
      TADD(v375, v371, v373);
      // pto: %t__tmp_v190
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v377 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v190
      ;
      uint64_t v378 = (uint64_t) v43;
      TASSIGN(v377, v378);
      pipe_barrier(PIPE_V);
      TGATHER(v377, v375, v343, v323);
      // pto: %sum4_inline838__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v379 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %sum4_inline838__ssa_v1
      ;
      uint64_t v380 = (uint64_t) v42;
      TASSIGN(v379, v380);
      pipe_barrier(PIPE_V);
      TADD(v379, v375, v377);
      // pto: %t__tmp_v191
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v381 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v191
      ;
      uint64_t v382 = (uint64_t) v44;
      TASSIGN(v381, v382);
      pipe_barrier(PIPE_V);
      TDIV(v381, v371, v379);
      // pto: %probs_inline794__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v383 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %probs_inline794__ssa_v1
      ;
      uint64_t v384 = (uint64_t) v44;
      TASSIGN(v383, v384);
      pipe_barrier(PIPE_V);
      TADDS(v383, v381, v80);
      // pto: %t__tmp_v192
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v385 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v192
      ;
      uint64_t v386 = (uint64_t) v42;
      TASSIGN(v385, v386);
      pipe_barrier(PIPE_V);
      TGATHER(v385, v383, v351, v323);
      // pto: %col2_inline861__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v387 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %col2_inline861__ssa_v4
      ;
      uint64_t v388 = (uint64_t) v42;
      TASSIGN(v387, v388);
      pipe_barrier(PIPE_V);
      TADD(v387, v383, v385);
      // pto: %t__tmp_v193
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v389 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v193
      ;
      uint64_t v390 = (uint64_t) v43;
      TASSIGN(v389, v390);
      pipe_barrier(PIPE_V);
      TGATHER(v389, v387, v359, v323);
      // pto: %t__tmp_v194
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v391 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v194
      ;
      uint64_t v392 = (uint64_t) v42;
      TASSIGN(v391, v392);
      pipe_barrier(PIPE_V);
      TADD(v391, v387, v389);
      // pto: %col4_inline869__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v393 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %col4_inline869__ssa_v4
      ;
      uint64_t v394 = (uint64_t) v42;
      TASSIGN(v393, v394);
      pipe_barrier(PIPE_V);
      TADDS(v393, v391, v80);
      // pto: %cur_inline703__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v395 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %cur_inline703__ssa_v4
      ;
      uint64_t v396 = (uint64_t) v44;
      TASSIGN(v395, v396);
      pipe_barrier(PIPE_V);
      TDIV(v395, v383, v393);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      for (int64_t v397 = v74; v397 < v88; v397 += v64) {
        // pto: %t__tmp_v195
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v398 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %t__tmp_v195
        ;
        uint64_t v399 = (uint64_t) v42;
        TASSIGN(v398, v399);
        pipe_barrier(PIPE_V);
        TGATHER(v398, v395, v335, v323);
        // pto: %row2_inline873__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v400 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %row2_inline873__ssa_v1
        ;
        uint64_t v401 = (uint64_t) v42;
        TASSIGN(v400, v401);
        pipe_barrier(PIPE_V);
        TADD(v400, v395, v398);
        // pto: %t__tmp_v196
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v402 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %t__tmp_v196
        ;
        uint64_t v403 = (uint64_t) v43;
        TASSIGN(v402, v403);
        pipe_barrier(PIPE_V);
        TGATHER(v402, v400, v343, v323);
        // pto: %t__tmp_v197
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v404 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %t__tmp_v197
        ;
        uint64_t v405 = (uint64_t) v42;
        TASSIGN(v404, v405);
        pipe_barrier(PIPE_V);
        TADD(v404, v400, v402);
        // pto: %row4_inline864__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v406 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %row4_inline864__ssa_v1
        ;
        uint64_t v407 = (uint64_t) v42;
        TASSIGN(v406, v407);
        pipe_barrier(PIPE_V);
        TADDS(v406, v404, v80);
        // pto: %norm_inline875__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v408 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %norm_inline875__ssa_v1
        ;
        uint64_t v409 = (uint64_t) v42;
        TASSIGN(v408, v409);
        pipe_barrier(PIPE_V);
        TDIV(v408, v395, v406);
        // pto: %t__tmp_v198
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v410 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %t__tmp_v198
        ;
        uint64_t v411 = (uint64_t) v43;
        TASSIGN(v410, v411);
        pipe_barrier(PIPE_V);
        TGATHER(v410, v408, v351, v323);
        // pto: %col2_inline861__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v412 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %col2_inline861__ssa_v7
        ;
        uint64_t v413 = (uint64_t) v43;
        TASSIGN(v412, v413);
        pipe_barrier(PIPE_V);
        TADD(v412, v408, v410);
        // pto: %t__tmp_v199
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v414 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %t__tmp_v199
        ;
        uint64_t v415 = (uint64_t) v38;
        TASSIGN(v414, v415);
        pipe_barrier(PIPE_V);
        TGATHER(v414, v412, v359, v323);
        // pto: %t__tmp_v200
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v416 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %t__tmp_v200
        ;
        uint64_t v417 = (uint64_t) v43;
        TASSIGN(v416, v417);
        pipe_barrier(PIPE_V);
        TADD(v416, v412, v414);
        // pto: %col4_inline869__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v418 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %col4_inline869__ssa_v7
        ;
        uint64_t v419 = (uint64_t) v43;
        TASSIGN(v418, v419);
        pipe_barrier(PIPE_V);
        TADDS(v418, v416, v80);
        // pto: %cur_inline703__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v420 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %cur_inline703__ssa_v7
        ;
        uint64_t v421 = (uint64_t) v44;
        TASSIGN(v420, v421);
        pipe_barrier(PIPE_V);
        TDIV(v420, v408, v418);
      };
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      // pto: %t__tmp_v201
      ;
      Tile<TileType::Vec, float, 4, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v422 = Tile<TileType::Vec, float, 4, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v77, v69);
      // pto: %t__tmp_v201
      ;
      uint64_t v423 = (uint64_t) v44;
      TASSIGN(v422, v423);
      v422.SetValidShape(v240, v69);
      // pto: %comb_inline765__ssa_v0_pview
      ;
      const int64_t v424 = 0;
      // pto: %comb_inline765__ssa_v0_pview
      ;
      __gm__ float* v425 = PTOAS__GLOBAL_TENSOR_DATA(v145);
      // pto: %comb_inline765__ssa_v0_pview
      ;
      const int64_t v426 = 1;
      // pto: %comb_inline765__ssa_v0_pview
      ;
      const int64_t v427 = 1;
      // pto: %comb_inline765__ssa_v0_pview
      ;
      const int64_t v428 = 1;
      // pto: %comb_inline765__ssa_v0_pview
      ;
      int64_t v429 = v240 * v69;
      // pto: %comb_inline765__ssa_v0_pview
      ;
      int64_t v430 = v428 * v429;
      // pto: %comb_inline765__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, 16> v431 = pto::Shape<1, 1, 1, -1, 16>(v426, v427, v428, v240, v69);
      // pto: %comb_inline765__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v432 = pto::Stride<-1, -1, -1, -1, -1>(v427 * v430, v430, v429, v69, v64);
      // pto: %comb_inline765__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v433 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v425 + (v424 + v264 * v69 + v74 * v64), v431, v432);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      TSTORE(v433, v422);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    };
  } else {
    // pto: %74
    ;
    int64_t v434 = (int64_t) ((uint64_t) v230 - (uint64_t) v18);
    // pto: %76
    ;
    int64_t v435 = v434 < v22 ? v434 : v22;
    // pto: %75, %77, %78
    ;
    int64_t v436 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v434 * (uint64_t) v21) + (uint64_t) v435) * (uint64_t) v75);
    // pto: %79
    ;
    int64_t v437 = (int64_t) ((uint64_t) v22 - (uint64_t) v434);
    // pto: %80
    ;
    int64_t v438 = v437 < v74 ? v74 : v437;
    // pto: %81
    ;
    int64_t v439 = v438 < v64 ? v438 : v64;
    // pto: %84, %82, %83
    ;
    int64_t v440 = (int64_t) ((uint64_t) v436 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v21 + (uint64_t) v439) * (uint64_t) v75));
    // pto: %85
    ;
    // pto: %86
    ;
    // pto: %d_begin_inline766__phi_v2
    ;
    int64_t v441;
    // pto: %work_begin_inline778__phi_v3
    ;
    int64_t v442;
    // pto: %work_end_inline767__phi_v3
    ;
    int64_t v443;
    if (v24 < v25) {
      // pto: %87
      ;
      // pto: %d_worker_inline932__phi_v2
      ;
      int64_t v444;
      // pto: %d_workers_inline782__phi_v2
      ;
      int64_t v445;
      // pto: %pair_id_inline816__phi_v2
      ;
      int64_t v446;
      if (v434 < v26) {
        // pto: %88
        ;
        int64_t v447 = (int64_t) ((uint64_t) v27 + (uint64_t) v64);
        // pto: %89
        ;
        // pto: %91
        ;
        v444 = v434 % v447;
        v445 = v447;
        v446 = v434 / v447;
      } else {
        // pto: %93
        ;
        int64_t v448 = (int64_t) ((uint64_t) v434 - (uint64_t) v26);
        // pto: %95, %94
        ;
        // pto: %96
        ;
        v444 = v448 % v27;
        v445 = v27;
        v446 = (int64_t) ((uint64_t) v28 + (uint64_t) (v448 / v27));
      };
      // pto: %97
      ;
      uint64_t v449 = (uint64_t) v446;
      // pto: %97
      ;
      int64_t v450 = (int64_t) (v449 * (uint64_t) v75);
      // pto: %98
      ;
      int64_t v451 = v23 < v450 ? v23 : v450;
      // pto: %99
      ;
      int64_t v452 = (int64_t) ((uint64_t) v451 + (uint64_t) v75);
      // pto: %100
      ;
      // pto: %101
      ;
      int64_t v453 = v68 / v445;
      // pto: %102
      ;
      int64_t v454 = v68 % v445;
      // pto: %103
      ;
      uint64_t v455 = (uint64_t) v444;
      // pto: %104
      ;
      bool v456 = v444 < v454;
      // pto: %104
      ;
      int64_t v457 = v456 ? v444 : v454;
      // pto: %103, %105
      ;
      v441 = (int64_t) ((uint64_t) ((int64_t) v455 * (uint64_t) v453) + (uint64_t) v457);
      v442 = v451;
      v443 = v23 < v452 ? v23 : v452;
    } else {
      v441 = v74;
      v442 = v436;
      v443 = v23 < v440 ? v23 : v440;
    };
    // pto: %106
    ;
    bool v458 = v25 <= v434;
    // pto: %work_begin_inline778__phi_v5
    ;
    int64_t v459 = v458 ? v74 : v442;
    // pto: %work_end_inline767__phi_v5
    ;
    int64_t v460 = v458 ? v74 : v443;
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    for (int64_t v461 = v459; v461 < v460; v461 += v77) {
      // pto: %107
      ;
      int64_t v462 = (int64_t) ((uint64_t) v460 - (uint64_t) v461);
      // pto: %108
      ;
      int64_t v463 = v462 < v77 ? v462 : v77;
      // pto: %109
      ;
      int64_t v464 = (int64_t) ((uint64_t) v463 * (uint64_t) v63);
      // pto: %m_parts_load_inline942__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v465 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v464);
      // pto: %m_parts_load_inline942__ssa_v0
      ;
      uint64_t v466 = (uint64_t) v36;
      TASSIGN(v465, v466);
      // pto: %110
      ;
      int64_t v467 = (int64_t) ((uint64_t) v461 * (uint64_t) v63);
      // pto: %111
      ;
      int64_t v468 = v467 < v74 ? v74 : v467;
      // pto: %113
      ;
      const int64_t v469 = 0;
      // pto: %113
      ;
      __gm__ float* v470 = PTOAS__GLOBAL_TENSOR_DATA(v119);
      // pto: %113
      ;
      const int64_t v471 = 1;
      // pto: %113
      ;
      const int64_t v472 = 1;
      // pto: %113
      ;
      const int64_t v473 = 1;
      // pto: %113
      ;
      int64_t v474 = v68 * v111;
      // pto: %113
      ;
      int64_t v475 = v473 * v474;
      // pto: %113
      ;
      pto::Shape<1, 1, 1, 8, -1> v476 = pto::Shape<1, 1, 1, 8, -1>(v471, v472, v473, v68, v464);
      // pto: %113
      ;
      pto::Stride<-1, -1, -1, -1, -1> v477 = pto::Stride<-1, -1, -1, -1, -1>(v472 * v475, v475, v474, v111, v64);
      // pto: %113
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v478 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v470 + (v469 + v74 * v111 + v468 * v64), v476, v477);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      TLOAD(v465, v478);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      // pto: %t__tmp_v202
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v479 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v68, v78);
      // pto: %t__tmp_v202
      ;
      uint64_t v480 = (uint64_t) v36;
      TASSIGN(v479, v480);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      TFILLPAD<pto::TFillPadMode::InPlace>(v479, v465);
      v479.SetValidShape(v68, v78);
      // pto: %t__tmp_v203
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v481 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v78);
      // pto: %t__tmp_v203
      ;
      uint64_t v482 = (uint64_t) v36;
      TASSIGN(v481, v482);
      pipe_barrier(PIPE_V);
      TCOLSUM(v481, v479);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      // pto: %m_total_inline771__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v483 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v63);
      // pto: %m_total_inline771__ssa_v0
      ;
      uint64_t v484 = (uint64_t) v36;
      TASSIGN(v483, v484);
      // pto: %m_sq_load_inline911__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v485 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v463);
      // pto: %m_sq_load_inline911__ssa_v0
      ;
      uint64_t v486 = (uint64_t) v38;
      TASSIGN(v485, v486);
      // pto: %114
      ;
      int64_t v487 = v461 < v74 ? v74 : v461;
      // pto: %115
      ;
      int64_t v488 = (int64_t) v29;
      // pto: %115
      ;
      const int64_t v489 = 0;
      // pto: %115
      ;
      __gm__ float* v490 = PTOAS__GLOBAL_TENSOR_DATA(v128);
      // pto: %115
      ;
      const int64_t v491 = 1;
      // pto: %115
      ;
      const int64_t v492 = 1;
      // pto: %115
      ;
      const int64_t v493 = 1;
      // pto: %115
      ;
      int64_t v494 = v68 * v488;
      // pto: %115
      ;
      int64_t v495 = v493 * v494;
      // pto: %115
      ;
      pto::Shape<1, 1, 1, 8, -1> v496 = pto::Shape<1, 1, 1, 8, -1>(v491, v492, v493, v68, v463);
      // pto: %115
      ;
      pto::Stride<-1, -1, -1, -1, -1> v497 = pto::Stride<-1, -1, -1, -1, -1>(v492 * v495, v495, v494, v488, v64);
      // pto: %115
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v498 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v490 + (v489 + v74 * v488 + v487 * v64), v496, v497);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      TLOAD(v485, v498);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      // pto: %t__tmp_v204
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v499 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v68, v68);
      // pto: %t__tmp_v204
      ;
      uint64_t v500 = (uint64_t) v38;
      TASSIGN(v499, v500);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      TFILLPAD<pto::TFillPadMode::InPlace>(v499, v485);
      v499.SetValidShape(v68, v68);
      // pto: %m_rms_tmp_inline853__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v501 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %m_rms_tmp_inline853__ssa_v0
      ;
      uint64_t v502 = (uint64_t) v39;
      TASSIGN(v501, v502);
      // pto: %t__tmp_v205
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v503 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v205
      ;
      uint64_t v504 = (uint64_t) v38;
      TASSIGN(v503, v504);
      pipe_barrier(PIPE_V);
      TCOLSUM(v503, v499);
      // pto: %t__tmp_v206
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v505 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v206
      ;
      uint64_t v506 = (uint64_t) v38;
      TASSIGN(v505, v506);
      pipe_barrier(PIPE_V);
      TMULS(v505, v503, v79);
      // pto: %t__tmp_v207
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v507 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v207
      ;
      uint64_t v508 = (uint64_t) v38;
      TASSIGN(v507, v508);
      pipe_barrier(PIPE_V);
      TADDS(v507, v505, v80);
      // pto: %t__tmp_v208
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v509 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v208
      ;
      uint64_t v510 = (uint64_t) v46;
      TASSIGN(v509, v510);
      pipe_barrier(PIPE_V);
      TRSQRT(v509, v507, v501);
      // pto: %m_inv_inline946__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v511 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v64);
      // pto: %m_inv_inline946__ssa_v0
      ;
      uint64_t v512 = (uint64_t) v46;
      TASSIGN(v511, v512);
      // pto: %gate_base_inline675__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v513 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %gate_base_inline675__ssa_v0
      ;
      uint64_t v514 = (uint64_t) v49;
      TASSIGN(v513, v514);
      // pto: %116
      ;
      __gm__ float* v515 = PTOAS__GLOBAL_TENSOR_DATA(v136);
      // pto: %116
      ;
      pto::Shape<1, 1, 1, 1, 8> v516 = pto::Shape<1, 1, 1, 1, 8>();
      // pto: %116
      ;
      pto::Stride<24, 24, 24, 24, 1> v517 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %116
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v518 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v515, v516, v517);
      TLOAD(v513, v518);
      // pto: %gate_scales_inline859__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v519 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v67);
      // pto: %gate_scales_inline859__ssa_v0
      ;
      uint64_t v520 = (uint64_t) v50;
      TASSIGN(v519, v520);
      // pto: %hc_scale_2d_inline775__ssa_v0_pview
      ;
      __gm__ float* v521 = PTOAS__GLOBAL_TENSOR_DATA(v153);
      // pto: %hc_scale_2d_inline775__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 3> v522 = pto::Shape<1, 1, 1, 1, 3>();
      // pto: %hc_scale_2d_inline775__ssa_v0_pview
      ;
      pto::Stride<3, 3, 3, 3, 1> v523 = pto::Stride<3, 3, 3, 3, 1>();
      // pto: %hc_scale_2d_inline775__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND> v524 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND>(v521, v522, v523);
      TLOAD(v519, v524);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      // pto: %t__ci_tmp_v1
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v525 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v82);
      // pto: %t__ci_tmp_v1
      ;
      uint64_t v526 = (uint64_t) v38;
      TASSIGN(v525, v526);
      // pto: %t__tmp_v209
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v527 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v209
      ;
      uint64_t v528 = (uint64_t) v51;
      TASSIGN(v527, v528);
      // pto: %117
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v529;
      TRESHAPE(v529, v525);
      // pto: %118
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v530;
      TRESHAPE(v530, v527);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v530, v71, v529);
      // pto: %gate_ids_inline947__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v531 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %gate_ids_inline947__ssa_v0
      ;
      uint64_t v532 = (uint64_t) v38;
      TASSIGN(v531, v532);
      pipe_barrier(PIPE_V);
      RoundMode v533 = RoundMode::CAST_ROUND;
      SaturationMode v534 = SaturationMode::ON;
      TCVT(v531, v527, v533, v534);
      // pto: %t__tmp_v210
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v535 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v210
      ;
      uint64_t v536 = (uint64_t) v38;
      TASSIGN(v535, v536);
      pipe_barrier(PIPE_V);
      TMULS(v535, v531, v89);
      // pto: %gate_scale_ids_inline897__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v537 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %gate_scale_ids_inline897__ssa_v0
      ;
      uint64_t v538 = (uint64_t) v38;
      TASSIGN(v537, v538);
      pipe_barrier(PIPE_V);
      RoundMode v539 = RoundMode::CAST_FLOOR;
      SaturationMode v540 = SaturationMode::ON;
      TCVT(v537, v535, v539, v540);
      // pto: %gate_tmp_inline914__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v541 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %gate_tmp_inline914__ssa_v0
      ;
      uint64_t v542 = (uint64_t) v52;
      TASSIGN(v541, v542);
      // pto: %gate_scale_vec_inline832__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v543 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %gate_scale_vec_inline832__ssa_v0
      ;
      uint64_t v544 = (uint64_t) v40;
      TASSIGN(v543, v544);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      TGATHER(v543, v519, v537, v541);
      // pto: %t__tmp_v211
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v545 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v68);
      // pto: %t__tmp_v211
      ;
      uint64_t v546 = (uint64_t) v36;
      TASSIGN(v545, v546);
      // pto: %119
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v547;
      // pto: %119
      ;
      uint64_t v548 = (uint64_t) v36;
      TASSIGN(v547, v548);
      // pto: %gate_raw_inline948__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v549 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v68);
      // pto: %gate_raw_inline948__ssa_v0
      ;
      uint64_t v550 = (uint64_t) v36;
      TASSIGN(v549, v550);
      TADDS(v549, v547, v73);
      // pto: %gate_norm_inline774__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v551 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v68);
      // pto: %gate_norm_inline774__ssa_v0
      ;
      uint64_t v552 = (uint64_t) v36;
      TASSIGN(v551, v552);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v551, v549, v511);
      // pto: %gate_scaled_inline949__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v553 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v68);
      // pto: %gate_scaled_inline949__ssa_v0
      ;
      uint64_t v554 = (uint64_t) v36;
      TASSIGN(v553, v554);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v553, v551, v543);
      // pto: %t__tmp_v212
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v555 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v68);
      // pto: %t__tmp_v212
      ;
      uint64_t v556 = (uint64_t) v44;
      TASSIGN(v555, v556);
      TCOLEXPAND(v555, v513);
      // pto: %gate_logits_inline951__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v557 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v68);
      // pto: %gate_logits_inline951__ssa_v0
      ;
      uint64_t v558 = (uint64_t) v36;
      TASSIGN(v557, v558);
      pipe_barrier(PIPE_V);
      TADD(v557, v553, v555);
      // pto: %t__tmp_v213
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v559 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v68);
      // pto: %t__tmp_v213
      ;
      uint64_t v560 = (uint64_t) v36;
      TASSIGN(v559, v560);
      pipe_barrier(PIPE_V);
      TNEG(v559, v557);
      // pto: %t__tmp_v214
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v561 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v68);
      // pto: %t__tmp_v214
      ;
      uint64_t v562 = (uint64_t) v36;
      TASSIGN(v561, v562);
      pipe_barrier(PIPE_V);
      TEXP(v561, v559);
      // pto: %t__tmp_v215
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v563 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v68);
      // pto: %t__tmp_v215
      ;
      uint64_t v564 = (uint64_t) v36;
      TASSIGN(v563, v564);
      pipe_barrier(PIPE_V);
      TADDS(v563, v561, v90);
      // pto: %gate_sigmoid_inline709__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v565 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v68);
      // pto: %gate_sigmoid_inline709__ssa_v0
      ;
      uint64_t v566 = (uint64_t) v53;
      TASSIGN(v565, v566);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
      TRECIP(v565, v563);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      // pto: %t__ci_tmp_v2
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v567 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v82);
      // pto: %t__ci_tmp_v2
      ;
      uint64_t v568 = (uint64_t) v36;
      TASSIGN(v567, v568);
      // pto: %t__tmp_v216
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v569 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v216
      ;
      uint64_t v570 = (uint64_t) v43;
      TASSIGN(v569, v570);
      // pto: %120
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v571;
      TRESHAPE(v571, v567);
      // pto: %121
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v572;
      TRESHAPE(v572, v569);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v572, v71, v571);
      // pto: %post_ids_float_inline953__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v573 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %post_ids_float_inline953__ssa_v0
      ;
      uint64_t v574 = (uint64_t) v36;
      TASSIGN(v573, v574);
      pipe_barrier(PIPE_V);
      RoundMode v575 = RoundMode::CAST_ROUND;
      SaturationMode v576 = SaturationMode::ON;
      TCVT(v573, v569, v575, v576);
      // pto: %t__tmp_v217
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v577 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v217
      ;
      uint64_t v578 = (uint64_t) v54;
      TASSIGN(v577, v578);
      pipe_barrier(PIPE_V);
      TMULS(v577, v573, v89);
      // pto: %t__tmp_v218
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v579 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v218
      ;
      uint64_t v580 = (uint64_t) v54;
      TASSIGN(v579, v580);
      pipe_barrier(PIPE_V);
      RoundMode v581 = RoundMode::CAST_FLOOR;
      SaturationMode v582 = SaturationMode::ON;
      TCVT(v579, v577, v581, v582);
      // pto: %post_rows_inline823__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v583 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %post_rows_inline823__ssa_v0
      ;
      uint64_t v584 = (uint64_t) v54;
      TASSIGN(v583, v584);
      pipe_barrier(PIPE_V);
      RoundMode v585 = RoundMode::CAST_ROUND;
      SaturationMode v586 = SaturationMode::ON;
      TCVT(v583, v579, v585, v586);
      // pto: %t__tmp_v219
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v587 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v219
      ;
      uint64_t v588 = (uint64_t) v54;
      TASSIGN(v587, v588);
      pipe_barrier(PIPE_V);
      TMULS(v587, v583, v91);
      // pto: %t__tmp_v220
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v589 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v220
      ;
      uint64_t v590 = (uint64_t) v36;
      TASSIGN(v589, v590);
      pipe_barrier(PIPE_V);
      TADD(v589, v573, v587);
      // pto: %t__tmp_v221
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v591 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v221
      ;
      uint64_t v592 = (uint64_t) v36;
      TASSIGN(v591, v592);
      pipe_barrier(PIPE_V);
      TADDS(v591, v589, v91);
      // pto: %post_ids_inline672__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v593 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %post_ids_inline672__ssa_v0
      ;
      uint64_t v594 = (uint64_t) v36;
      TASSIGN(v593, v594);
      pipe_barrier(PIPE_V);
      RoundMode v595 = RoundMode::CAST_ROUND;
      SaturationMode v596 = SaturationMode::ON;
      TCVT(v593, v591, v595, v596);
      // pto: %post_tmp_inline671__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v597 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %post_tmp_inline671__ssa_v0
      ;
      uint64_t v598 = (uint64_t) v54;
      TASSIGN(v597, v598);
      // pto: %t__tmp_v222
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v599 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v222
      ;
      uint64_t v600 = (uint64_t) v53;
      TASSIGN(v599, v600);
      // pto: %post_values_inline843__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v601 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %post_values_inline843__ssa_v0
      ;
      uint64_t v602 = (uint64_t) v44;
      TASSIGN(v601, v602);
      pipe_barrier(PIPE_V);
      TGATHER(v601, v599, v593, v597);
      // pto: %t__tmp_v223
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v603 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v223
      ;
      uint64_t v604 = (uint64_t) v36;
      TASSIGN(v603, v604);
      pipe_barrier(PIPE_V);
      TMULS(v603, v601, v92);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      // pto: %122
      ;
      int64_t v605 = (int64_t) ((uint64_t) v463 * (uint64_t) v77);
      v603.SetValidShape(v64, v605);
      // pto: %123
      ;
      bool v606 = v441 == v74;
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      if (v606) {
        // pto: %125
        ;
        int64_t v607 = (int64_t) ((uint64_t) v461 * (uint64_t) v77);
        // pto: %126
        ;
        int64_t v608 = v607 < v74 ? v74 : v607;
        // pto: %post_flat_inline909__ssa_v0_pview
        ;
        const int64_t v609 = 0;
        // pto: %post_flat_inline909__ssa_v0_pview
        ;
        __gm__ float* v610 = PTOAS__GLOBAL_TENSOR_DATA(v162);
        // pto: %post_flat_inline909__ssa_v0_pview
        ;
        const int64_t v611 = 1;
        // pto: %post_flat_inline909__ssa_v0_pview
        ;
        const int64_t v612 = 1;
        // pto: %post_flat_inline909__ssa_v0_pview
        ;
        const int64_t v613 = 1;
        // pto: %post_flat_inline909__ssa_v0_pview
        ;
        int64_t v614 = v64 * v154;
        // pto: %post_flat_inline909__ssa_v0_pview
        ;
        int64_t v615 = v613 * v614;
        // pto: %post_flat_inline909__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, -1> v616 = pto::Shape<1, 1, 1, 1, -1>(v611, v612, v613, v64, v605);
        // pto: %post_flat_inline909__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v617 = pto::Stride<-1, -1, -1, -1, -1>(v612 * v615, v615, v614, v154, v64);
        // pto: %post_flat_inline909__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v618 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v610 + (v609 + v74 * v154 + v608 * v64), v616, v617);
        TSTORE(v618, v603);
      };
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      // pto: %127
      ;
      int64_t v619 = v463 / v75;
      // pto: %128
      ;
      int64_t v620 = (int64_t) ((uint64_t) v619 * (uint64_t) v75);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      for (int64_t v621 = v74; v621 < v620; v621 += v75) {
        // pto: %129
        ;
        int64_t v622 = (int64_t) ((uint64_t) v461 + (uint64_t) v621);
        // pto: %130
        ;
        int64_t v623 = (int64_t) ((uint64_t) v621 + (uint64_t) v64);
        // pto: %131
        ;
        int64_t v624 = (int64_t) ((uint64_t) v461 + (uint64_t) v623);
        // pto: %flat_offset_mul
        ;
        int64_t v625 = (int64_t) ((uint64_t) v621 * (uint64_t) v68);
        // pto: %t__tmp_v224
        ;
        float v626 = v565.GetValue(v625);
        // pto: %132
        ;
        float v627 = v626 + v80;
        // pto: %134, %t__tmp_v225
        ;
        float v628 = v565.GetValue((int64_t) ((uint64_t) v625 + (uint64_t) v64));
        // pto: %135
        ;
        float v629 = v628 + v80;
        // pto: %137, %t__tmp_v226
        ;
        float v630 = v565.GetValue((int64_t) ((uint64_t) v625 + (uint64_t) v75));
        // pto: %138
        ;
        float v631 = v630 + v80;
        // pto: %140, %t__tmp_v227
        ;
        float v632 = v565.GetValue((int64_t) ((uint64_t) v625 + (uint64_t) v67));
        // pto: %141
        ;
        float v633 = v632 + v80;
        // pto: %t__tmp_v228
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v634 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %t__tmp_v228
        ;
        uint64_t v635 = (uint64_t) v55;
        TASSIGN(v634, v635);
        // pto: %142
        ;
        int64_t v636 = v622 < v74 ? v74 : v622;
        // pto: %143
        ;
        __gm__ bfloat16_t* v637 = PTOAS__GLOBAL_TENSOR_DATA(v110);
        // pto: %143
        ;
        const int64_t v638 = 0;
        // pto: %143
        ;
        const int64_t v639 = 16384;
        // pto: %143
        ;
        pto::Shape<1, 1, 1, 1, 4096> v640 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %143
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v641 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %143
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v642 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v637 + (v638 + v636 * v639), v640, v641);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
        TLOAD(v634, v642);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        // pto: %t__tmp_v229
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v643 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %t__tmp_v229
        ;
        uint64_t v644 = (uint64_t) v56;
        TASSIGN(v643, v644);
        // pto: %145
        ;
        __gm__ bfloat16_t* v645 = PTOAS__GLOBAL_TENSOR_DATA(v110);
        // pto: %145
        ;
        const int64_t v646 = 4096;
        // pto: %145
        ;
        const int64_t v647 = 16384;
        // pto: %145
        ;
        pto::Shape<1, 1, 1, 1, 4096> v648 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %145
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v649 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %145
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v650 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v645 + (v646 + v636 * v647), v648, v649);
        TLOAD(v643, v650);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %t__tmp_v230
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v651 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %t__tmp_v230
        ;
        uint64_t v652 = (uint64_t) v57;
        TASSIGN(v651, v652);
        // pto: %147
        ;
        __gm__ bfloat16_t* v653 = PTOAS__GLOBAL_TENSOR_DATA(v110);
        // pto: %147
        ;
        const int64_t v654 = 8192;
        // pto: %147
        ;
        const int64_t v655 = 16384;
        // pto: %147
        ;
        pto::Shape<1, 1, 1, 1, 4096> v656 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %147
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v657 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %147
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v658 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v653 + (v654 + v636 * v655), v656, v657);
        TLOAD(v651, v658);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %t__tmp_v231
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v659 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %t__tmp_v231
        ;
        uint64_t v660 = (uint64_t) v58;
        TASSIGN(v659, v660);
        // pto: %149
        ;
        __gm__ bfloat16_t* v661 = PTOAS__GLOBAL_TENSOR_DATA(v110);
        // pto: %149
        ;
        const int64_t v662 = 12288;
        // pto: %149
        ;
        const int64_t v663 = 16384;
        // pto: %149
        ;
        pto::Shape<1, 1, 1, 1, 4096> v664 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %149
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v665 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %149
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v666 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v661 + (v662 + v636 * v663), v664, v665);
        TLOAD(v659, v666);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %152
        ;
        int64_t v667 = (int64_t) ((uint64_t) v623 * (uint64_t) v68);
        // pto: %150
        ;
        float v668 = v565.GetValue(v667);
        // pto: %154
        ;
        float v669 = v668 + v80;
        // pto: %158, %155
        ;
        float v670 = v565.GetValue((int64_t) ((uint64_t) v667 + (uint64_t) v64));
        // pto: %159
        ;
        float v671 = v670 + v80;
        // pto: %163, %160
        ;
        float v672 = v565.GetValue((int64_t) ((uint64_t) v667 + (uint64_t) v75));
        // pto: %164
        ;
        float v673 = v672 + v80;
        // pto: %168, %165
        ;
        float v674 = v565.GetValue((int64_t) ((uint64_t) v667 + (uint64_t) v67));
        // pto: %169
        ;
        float v675 = v674 + v80;
        // pto: %4
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v676 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %4
        ;
        uint64_t v677 = (uint64_t) v35;
        TASSIGN(v676, v677);
        // pto: %170
        ;
        int64_t v678 = v624 < v74 ? v74 : v624;
        // pto: %171
        ;
        __gm__ bfloat16_t* v679 = PTOAS__GLOBAL_TENSOR_DATA(v110);
        // pto: %171
        ;
        const int64_t v680 = 0;
        // pto: %171
        ;
        const int64_t v681 = 16384;
        // pto: %171
        ;
        pto::Shape<1, 1, 1, 1, 4096> v682 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %171
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v683 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %171
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v684 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v679 + (v680 + v678 * v681), v682, v683);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v676, v684);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %5
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v685 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %5
        ;
        uint64_t v686 = (uint64_t) v59;
        TASSIGN(v685, v686);
        // pto: %173
        ;
        __gm__ bfloat16_t* v687 = PTOAS__GLOBAL_TENSOR_DATA(v110);
        // pto: %173
        ;
        const int64_t v688 = 4096;
        // pto: %173
        ;
        const int64_t v689 = 16384;
        // pto: %173
        ;
        pto::Shape<1, 1, 1, 1, 4096> v690 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %173
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v691 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %173
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v692 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v687 + (v688 + v678 * v689), v690, v691);
        TLOAD(v685, v692);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        // pto: %6
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v693 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %6
        ;
        uint64_t v694 = (uint64_t) v60;
        TASSIGN(v693, v694);
        // pto: %175
        ;
        __gm__ bfloat16_t* v695 = PTOAS__GLOBAL_TENSOR_DATA(v110);
        // pto: %175
        ;
        const int64_t v696 = 8192;
        // pto: %175
        ;
        const int64_t v697 = 16384;
        // pto: %175
        ;
        pto::Shape<1, 1, 1, 1, 4096> v698 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %175
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v699 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %175
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v700 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v695 + (v696 + v678 * v697), v698, v699);
        TLOAD(v693, v700);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        // pto: %7
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v701 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %7
        ;
        uint64_t v702 = (uint64_t) v61;
        TASSIGN(v701, v702);
        // pto: %177
        ;
        __gm__ bfloat16_t* v703 = PTOAS__GLOBAL_TENSOR_DATA(v110);
        // pto: %177
        ;
        const int64_t v704 = 12288;
        // pto: %177
        ;
        const int64_t v705 = 16384;
        // pto: %177
        ;
        pto::Shape<1, 1, 1, 1, 4096> v706 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %177
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v707 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %177
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v708 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v703 + (v704 + v678 * v705), v706, v707);
        TLOAD(v701, v708);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        // pto: %scalar_x0_inline652__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v709 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %scalar_x0_inline652__ssa_v0
        ;
        uint64_t v710 = (uint64_t) v36;
        TASSIGN(v709, v710);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        RoundMode v711 = RoundMode::CAST_ROUND;
        SaturationMode v712 = SaturationMode::ON;
        TCVT(v709, v634, v711, v712);
        // pto: %scalar_y0_inline651__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v713 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %scalar_y0_inline651__ssa_v0
        ;
        uint64_t v714 = (uint64_t) v36;
        TASSIGN(v713, v714);
        pipe_barrier(PIPE_V);
        TMULS(v713, v709, v627);
        // pto: %scalar_x1_inline650__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v715 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %scalar_x1_inline650__ssa_v0
        ;
        uint64_t v716 = (uint64_t) v55;
        TASSIGN(v715, v716);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v717 = RoundMode::CAST_ROUND;
        SaturationMode v718 = SaturationMode::ON;
        TCVT(v715, v643, v717, v718);
        // pto: %scalar_y1_inline809__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v719 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %scalar_y1_inline809__ssa_v0
        ;
        uint64_t v720 = (uint64_t) v55;
        TASSIGN(v719, v720);
        pipe_barrier(PIPE_V);
        TMULS(v719, v715, v629);
        // pto: %scalar_sum01_inline649__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v721 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %scalar_sum01_inline649__ssa_v0
        ;
        uint64_t v722 = (uint64_t) v36;
        TASSIGN(v721, v722);
        pipe_barrier(PIPE_V);
        TADD(v721, v713, v719);
        // pto: %scalar_x2_inline787__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v723 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %scalar_x2_inline787__ssa_v0
        ;
        uint64_t v724 = (uint64_t) v55;
        TASSIGN(v723, v724);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        RoundMode v725 = RoundMode::CAST_ROUND;
        SaturationMode v726 = SaturationMode::ON;
        TCVT(v723, v651, v725, v726);
        // pto: %scalar_y2_inline648__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v727 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %scalar_y2_inline648__ssa_v0
        ;
        uint64_t v728 = (uint64_t) v55;
        TASSIGN(v727, v728);
        pipe_barrier(PIPE_V);
        TMULS(v727, v723, v631);
        // pto: %scalar_x3_inline647__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v729 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %scalar_x3_inline647__ssa_v0
        ;
        uint64_t v730 = (uint64_t) v56;
        TASSIGN(v729, v730);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        RoundMode v731 = RoundMode::CAST_ROUND;
        SaturationMode v732 = SaturationMode::ON;
        TCVT(v729, v659, v731, v732);
        // pto: %scalar_y3_inline796__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v733 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %scalar_y3_inline796__ssa_v0
        ;
        uint64_t v734 = (uint64_t) v56;
        TASSIGN(v733, v734);
        pipe_barrier(PIPE_V);
        TMULS(v733, v729, v633);
        // pto: %t__tmp_v232
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v735 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %t__tmp_v232
        ;
        uint64_t v736 = (uint64_t) v55;
        TASSIGN(v735, v736);
        pipe_barrier(PIPE_V);
        TADD(v735, v727, v733);
        // pto: %scalar_result_inline732__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v737 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %scalar_result_inline732__ssa_v0
        ;
        uint64_t v738 = (uint64_t) v36;
        TASSIGN(v737, v738);
        pipe_barrier(PIPE_V);
        TADD(v737, v721, v735);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
        // pto: %scalar_bf16_inline646__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v739 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %scalar_bf16_inline646__ssa_v0
        ;
        uint64_t v740 = (uint64_t) v36;
        TASSIGN(v739, v740);
        pipe_barrier(PIPE_V);
        RoundMode v741 = RoundMode::CAST_RINT;
        SaturationMode v742 = SaturationMode::ON;
        TCVT(v739, v737, v741, v742);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        // pto: %mixed_view_inline708__ssa_v0_pview
        ;
        __gm__ bfloat16_t* v743 = PTOAS__GLOBAL_TENSOR_DATA(v171);
        // pto: %mixed_view_inline708__ssa_v0_pview
        ;
        const int64_t v744 = 0;
        // pto: %mixed_view_inline708__ssa_v0_pview
        ;
        const int64_t v745 = 4096;
        // pto: %mixed_view_inline708__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, 4096> v746 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %mixed_view_inline708__ssa_v0_pview
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v747 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %mixed_view_inline708__ssa_v0_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v748 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v743 + (v744 + v636 * v745), v746, v747);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v748, v739);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        // pto: %8
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v749 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %8
        ;
        uint64_t v750 = (uint64_t) v62;
        TASSIGN(v749, v750);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        RoundMode v751 = RoundMode::CAST_ROUND;
        SaturationMode v752 = SaturationMode::ON;
        TCVT(v749, v676, v751, v752);
        // pto: %9
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v753 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %9
        ;
        uint64_t v754 = (uint64_t) v62;
        TASSIGN(v753, v754);
        pipe_barrier(PIPE_V);
        TMULS(v753, v749, v669);
        // pto: %10
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v755 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %10
        ;
        uint64_t v756 = (uint64_t) v35;
        TASSIGN(v755, v756);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        RoundMode v757 = RoundMode::CAST_ROUND;
        SaturationMode v758 = SaturationMode::ON;
        TCVT(v755, v685, v757, v758);
        // pto: %11
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v759 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %11
        ;
        uint64_t v760 = (uint64_t) v35;
        TASSIGN(v759, v760);
        pipe_barrier(PIPE_V);
        TMULS(v759, v755, v671);
        // pto: %12
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v761 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %12
        ;
        uint64_t v762 = (uint64_t) v62;
        TASSIGN(v761, v762);
        pipe_barrier(PIPE_V);
        TADD(v761, v753, v759);
        // pto: %13
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v763 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %13
        ;
        uint64_t v764 = (uint64_t) v35;
        TASSIGN(v763, v764);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        RoundMode v765 = RoundMode::CAST_ROUND;
        SaturationMode v766 = SaturationMode::ON;
        TCVT(v763, v693, v765, v766);
        // pto: %14
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v767 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %14
        ;
        uint64_t v768 = (uint64_t) v35;
        TASSIGN(v767, v768);
        pipe_barrier(PIPE_V);
        TMULS(v767, v763, v673);
        // pto: %15
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v769 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %15
        ;
        uint64_t v770 = (uint64_t) v59;
        TASSIGN(v769, v770);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        RoundMode v771 = RoundMode::CAST_ROUND;
        SaturationMode v772 = SaturationMode::ON;
        TCVT(v769, v701, v771, v772);
        // pto: %16
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v773 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %16
        ;
        uint64_t v774 = (uint64_t) v59;
        TASSIGN(v773, v774);
        pipe_barrier(PIPE_V);
        TMULS(v773, v769, v675);
        // pto: %17
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v775 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %17
        ;
        uint64_t v776 = (uint64_t) v35;
        TASSIGN(v775, v776);
        pipe_barrier(PIPE_V);
        TADD(v775, v767, v773);
        // pto: %18
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v777 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %18
        ;
        uint64_t v778 = (uint64_t) v62;
        TASSIGN(v777, v778);
        pipe_barrier(PIPE_V);
        TADD(v777, v761, v775);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %19
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v779 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %19
        ;
        uint64_t v780 = (uint64_t) v62;
        TASSIGN(v779, v780);
        pipe_barrier(PIPE_V);
        RoundMode v781 = RoundMode::CAST_RINT;
        SaturationMode v782 = SaturationMode::ON;
        TCVT(v779, v777, v781, v782);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        // pto: %181
        ;
        __gm__ bfloat16_t* v783 = PTOAS__GLOBAL_TENSOR_DATA(v171);
        // pto: %181
        ;
        const int64_t v784 = 0;
        // pto: %181
        ;
        const int64_t v785 = 4096;
        // pto: %181
        ;
        pto::Shape<1, 1, 1, 1, 4096> v786 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %181
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v787 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %181
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v788 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v783 + (v784 + v678 * v785), v786, v787);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v788, v779);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
      };
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_S, PIPE_V, EVENT_ID1);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      // pto: %182, %183
      ;
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      if ((int64_t) ((uint64_t) v463 - (uint64_t) v620) == v64) {
        // pto: %184
        ;
        int64_t v789 = (int64_t) ((uint64_t) v461 + (uint64_t) v620);
        int64_t v790 = (int64_t) ((uint64_t) v619 * (uint64_t) v69);
        // pto: %185
        ;
        float v791 = v565.GetValue(v790);
        // pto: %188
        ;
        float v792 = v791 + v80;
        // pto: %191, %189
        ;
        float v793 = v565.GetValue((int64_t) ((uint64_t) v790 + (uint64_t) v64));
        // pto: %192
        ;
        float v794 = v793 + v80;
        // pto: %195, %193
        ;
        float v795 = v565.GetValue((int64_t) ((uint64_t) v790 + (uint64_t) v75));
        // pto: %196
        ;
        float v796 = v795 + v80;
        // pto: %199, %197
        ;
        float v797 = v565.GetValue((int64_t) ((uint64_t) v790 + (uint64_t) v67));
        // pto: %200
        ;
        float v798 = v797 + v80;
        // pto: %20
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v799 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %20
        ;
        uint64_t v800 = (uint64_t) v55;
        TASSIGN(v799, v800);
        // pto: %201
        ;
        int64_t v801 = v789 < v74 ? v74 : v789;
        // pto: %202
        ;
        __gm__ bfloat16_t* v802 = PTOAS__GLOBAL_TENSOR_DATA(v110);
        // pto: %202
        ;
        const int64_t v803 = 0;
        // pto: %202
        ;
        const int64_t v804 = 16384;
        // pto: %202
        ;
        pto::Shape<1, 1, 1, 1, 4096> v805 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %202
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v806 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %202
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v807 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v802 + (v803 + v801 * v804), v805, v806);
        TLOAD(v799, v807);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %21
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v808 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %21
        ;
        uint64_t v809 = (uint64_t) v36;
        TASSIGN(v808, v809);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v810 = RoundMode::CAST_ROUND;
        SaturationMode v811 = SaturationMode::ON;
        TCVT(v808, v799, v810, v811);
        // pto: %22
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v812 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %22
        ;
        uint64_t v813 = (uint64_t) v36;
        TASSIGN(v812, v813);
        pipe_barrier(PIPE_V);
        TMULS(v812, v808, v792);
        // pto: %23
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v814 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %23
        ;
        uint64_t v815 = (uint64_t) v56;
        TASSIGN(v814, v815);
        // pto: %204
        ;
        __gm__ bfloat16_t* v816 = PTOAS__GLOBAL_TENSOR_DATA(v110);
        // pto: %204
        ;
        const int64_t v817 = 4096;
        // pto: %204
        ;
        const int64_t v818 = 16384;
        // pto: %204
        ;
        pto::Shape<1, 1, 1, 1, 4096> v819 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %204
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v820 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %204
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v821 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v816 + (v817 + v801 * v818), v819, v820);
        TLOAD(v814, v821);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %24
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v822 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %24
        ;
        uint64_t v823 = (uint64_t) v55;
        TASSIGN(v822, v823);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v824 = RoundMode::CAST_ROUND;
        SaturationMode v825 = SaturationMode::ON;
        TCVT(v822, v814, v824, v825);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %25
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v826 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %25
        ;
        uint64_t v827 = (uint64_t) v55;
        TASSIGN(v826, v827);
        pipe_barrier(PIPE_V);
        TMULS(v826, v822, v794);
        // pto: %26
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v828 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %26
        ;
        uint64_t v829 = (uint64_t) v36;
        TASSIGN(v828, v829);
        pipe_barrier(PIPE_V);
        TADD(v828, v812, v826);
        // pto: %27
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v830 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %27
        ;
        uint64_t v831 = (uint64_t) v56;
        TASSIGN(v830, v831);
        // pto: %206
        ;
        __gm__ bfloat16_t* v832 = PTOAS__GLOBAL_TENSOR_DATA(v110);
        // pto: %206
        ;
        const int64_t v833 = 8192;
        // pto: %206
        ;
        const int64_t v834 = 16384;
        // pto: %206
        ;
        pto::Shape<1, 1, 1, 1, 4096> v835 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %206
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v836 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %206
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v837 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v832 + (v833 + v801 * v834), v835, v836);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v830, v837);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %28
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v838 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %28
        ;
        uint64_t v839 = (uint64_t) v55;
        TASSIGN(v838, v839);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v840 = RoundMode::CAST_ROUND;
        SaturationMode v841 = SaturationMode::ON;
        TCVT(v838, v830, v840, v841);
        // pto: %29
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v842 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %29
        ;
        uint64_t v843 = (uint64_t) v55;
        TASSIGN(v842, v843);
        pipe_barrier(PIPE_V);
        TMULS(v842, v838, v796);
        // pto: %30
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v844 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %30
        ;
        uint64_t v845 = (uint64_t) v62;
        TASSIGN(v844, v845);
        // pto: %208
        ;
        __gm__ bfloat16_t* v846 = PTOAS__GLOBAL_TENSOR_DATA(v110);
        // pto: %208
        ;
        const int64_t v847 = 12288;
        // pto: %208
        ;
        const int64_t v848 = 16384;
        // pto: %208
        ;
        pto::Shape<1, 1, 1, 1, 4096> v849 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %208
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v850 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %208
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v851 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v846 + (v847 + v801 * v848), v849, v850);
        TLOAD(v844, v851);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %31
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v852 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %31
        ;
        uint64_t v853 = (uint64_t) v56;
        TASSIGN(v852, v853);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v854 = RoundMode::CAST_ROUND;
        SaturationMode v855 = SaturationMode::ON;
        TCVT(v852, v844, v854, v855);
        // pto: %32
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v856 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %32
        ;
        uint64_t v857 = (uint64_t) v56;
        TASSIGN(v856, v857);
        pipe_barrier(PIPE_V);
        TMULS(v856, v852, v798);
        // pto: %33
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v858 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %33
        ;
        uint64_t v859 = (uint64_t) v55;
        TASSIGN(v858, v859);
        pipe_barrier(PIPE_V);
        TADD(v858, v842, v856);
        // pto: %34
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v860 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %34
        ;
        uint64_t v861 = (uint64_t) v36;
        TASSIGN(v860, v861);
        pipe_barrier(PIPE_V);
        TADD(v860, v828, v858);
        // pto: %35
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v862 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %35
        ;
        uint64_t v863 = (uint64_t) v36;
        TASSIGN(v862, v863);
        pipe_barrier(PIPE_V);
        RoundMode v864 = RoundMode::CAST_RINT;
        SaturationMode v865 = SaturationMode::ON;
        TCVT(v862, v860, v864, v865);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %211
        ;
        __gm__ bfloat16_t* v866 = PTOAS__GLOBAL_TENSOR_DATA(v171);
        // pto: %211
        ;
        const int64_t v867 = 0;
        // pto: %211
        ;
        const int64_t v868 = 4096;
        // pto: %211
        ;
        pto::Shape<1, 1, 1, 1, 4096> v869 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %211
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v870 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %211
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v871 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v866 + (v867 + v801 * v868), v869, v870);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v871, v862);
      };
      set_flag(PIPE_S, PIPE_V, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    };
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  }
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}