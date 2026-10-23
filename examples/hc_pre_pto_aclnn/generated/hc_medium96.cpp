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
  // pto: %mixes_partials_inline790__ssa_v0_view
  const int64_t v51 = 1;
  // pto: %mixes_partials_inline790__ssa_v0_view
  const int64_t v52 = 1;
  // pto: %mixes_partials_inline790__ssa_v0_view
  const int64_t v53 = 1;
  // pto: %mixes_partials_inline790__ssa_v0_view
  int64_t v54 = v50 * v38;
  // pto: %mixes_partials_inline790__ssa_v0_view
  int64_t v55 = v53 * v54;
  // pto: %mixes_partials_inline790__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v56 = pto::Shape<1, 1, 1, -1, -1>(v51, v52, v53, v50, v38);
  // pto: %mixes_partials_inline790__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v57 = pto::Stride<-1, -1, -1, -1, -1>(v52 * v55, v55, v54, v38, v39);
  // pto: %mixes_partials_inline790__ssa_v0_view
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
  // pto: %mid_stream_product_inline718__phi_v10
  Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v67 = Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v38, v40);
  // pto: %mid_stream_product_inline718__phi_v10
  uint64_t v68 = (uint64_t) v35;
  TASSIGN(v67, v68);
  const int32_t v69 = 0;
  auto v70 = TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>(v13, v69, v34);
  // pto: %worker_inline766__ssa_v0
  // pto: %linear_workers_inline925__ssa_v0_idx
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
    // pto: %mid_stream_product_inline718__tile
    ;
    Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v74 = Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v38, v38);
    // pto: %mid_stream_product_inline718__tile
    ;
    uint64_t v75 = (uint64_t) v35;
    TASSIGN(v74, v75);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int64_t v76 = v44; v76 < v45; v76 += v39) {
      // pto: %13, %17, %16
      ;
      int64_t v77 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v72 * (uint64_t) v43) + (uint64_t) ((int64_t) (uint64_t) v76 * (uint64_t) v46));
      // pto: %mid_weight_chunk_inline710__tile
      ;
      Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v78 = Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v40, v46);
      // pto: %mid_weight_chunk_inline710__tile
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
      // pto: %mid_gathered_chunk_inline708__tile
      ;
      const int32_t v91 = 32;
      // pto: %mid_gathered_chunk_inline708__tile
      ;
      const int32_t v92 = 1024;
      // pto: %mid_gathered_chunk_inline708__tile
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
        // pto: %mid_weight_chunk_inline710__tile_t
        ;
        Tile<TileType::Mat, float, 1024, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v94 = Tile<TileType::Mat, float, 1024, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v46, v40);
        // pto: %mid_weight_chunk_inline710__tile_t
        ;
        uint64_t v95 = (uint64_t) v36;
        TASSIGN(v94, v95);
        // pto: %mid_stream_product_inline718__tile_l0_init_storage
        ;
        Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v38, v38);
        // pto: %mid_stream_product_inline718__tile_l0_init_storage
        ;
        uint64_t v97 = (uint64_t) v35;
        TASSIGN(v96, v97);
        v96.SetValidShape(v38, v40);
        for (int64_t v98 = v44; v98 < v46; v98 += v47) {
          // pto: %mid_stream_product_inline718__tile_l0_a
          ;
          Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v99 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v38, v48);
          // pto: %mid_stream_product_inline718__tile_l0_a
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
          // pto: %mid_stream_product_inline718__tile_l0_b
          ;
          Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v105 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v104, v40);
          // pto: %mid_stream_product_inline718__tile_l0_b
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
    // pto: %mixes_partials_inline790__iter_v1_pview
    ;
    __gm__ float* v147 = PTOAS__GLOBAL_TENSOR_DATA(v58);
    // pto: %mixes_partials_inline790__iter_v1_pview
    ;
    const int64_t v148 = 0;
    // pto: %mixes_partials_inline790__iter_v1_pview
    ;
    const int64_t v149 = 32;
    // pto: %mixes_partials_inline790__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 32, 32> v150 = pto::Shape<1, 1, 1, 32, 32>();
    // pto: %mixes_partials_inline790__iter_v1_pview
    ;
    pto::Stride<1024, 1024, 1024, 32, 1> v151 = pto::Stride<1024, 1024, 1024, 32, 1>();
    // pto: %mixes_partials_inline790__iter_v1_pview
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
  // pto: %c8256_i64
  const int64_t v38 = 8256;
  // pto: %c1088_i64
  const int64_t v39 = 1088;
  // pto: %c1344_i64
  const int64_t v40 = 1344;
  // pto: %c1600_i64
  const int64_t v41 = 1600;
  // pto: %c1152_i64
  const int64_t v42 = 1152;
  const int64_t v43 = 96;
  // pto: %c576_i64
  const int64_t v44 = 576;
  // pto: %c832_i64
  const int64_t v45 = 832;
  // pto: %c320_i64
  const int64_t v46 = 320;
  // pto: %c1856_i64
  const int64_t v47 = 1856;
  // pto: %c2112_i64
  const int64_t v48 = 2112;
  // pto: %c2368_i64
  const int64_t v49 = 2368;
  // pto: %c1888_i64
  const int64_t v50 = 1888;
  // pto: %c1920_i64
  const int64_t v51 = 1920;
  // pto: %c1952_i64
  const int64_t v52 = 1952;
  // pto: %c1120_i64
  const int64_t v53 = 1120;
  // pto: %c114752_i64
  const int64_t v54 = 114752;
  // pto: %c192_i64
  const int64_t v55 = 192;
  // pto: %c16448_i64
  const int64_t v56 = 16448;
  // pto: %c32832_i64
  const int64_t v57 = 32832;
  // pto: %c41024_i64
  const int64_t v58 = 41024;
  // pto: %c98368_i64
  const int64_t v59 = 98368;
  // pto: %c81984_i64
  const int64_t v60 = 81984;
  // pto: %c90176_i64
  const int64_t v61 = 90176;
  // pto: %c106560_i64
  const int64_t v62 = 106560;
  // pto: %c49216_i64
  const int64_t v63 = 49216;
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
  // pto: %c8_index
  const int64_t v69 = 8;
  const int64_t v70 = 16;
  // pto: %c4096_index
  const int64_t v71 = 4096;
  // pto: %c0_i32
  const int32_t v72 = 0;
  // pto: %c2048_index
  const int64_t v73 = 2048;
  // pto: %cst_40
  const float v74 = 0.0f;
  // pto: %c0_index
  const int64_t v75 = 0;
  // pto: %c2_index
  const int64_t v76 = 2;
  // pto: %c1024_index
  const int64_t v77 = 1024;
  // pto: %c4_index
  const int64_t v78 = 4;
  // pto: %c256_index
  const int64_t v79 = 256;
  // pto: %cst_46
  const float v80 = 6.10351563E-5f;
  // pto: %cst_47
  const float v81 = 9.99999997E-7f;
  // pto: %c64_index
  const int64_t v82 = 64;
  // pto: %c192_index
  const int64_t v83 = 192;
  // pto: %c128_index
  const int64_t v84 = 128;
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
  // pto: %221
  int64_t v94 = (int64_t) ((uint64_t) v29 * (uint64_t) v69);
  // pto: %rms_partials_inline825__ssa_v0_view
  const int64_t v95 = 1;
  // pto: %rms_partials_inline825__ssa_v0_view
  const int64_t v96 = 1;
  // pto: %rms_partials_inline825__ssa_v0_view
  const int64_t v97 = 1;
  // pto: %rms_partials_inline825__ssa_v0_view
  int64_t v98 = v94 * v65;
  // pto: %rms_partials_inline825__ssa_v0_view
  int64_t v99 = v97 * v98;
  // pto: %rms_partials_inline825__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v100 = pto::Shape<1, 1, 1, -1, -1>(v95, v96, v97, v94, v65);
  // pto: %rms_partials_inline825__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v101 = pto::Stride<-1, -1, -1, -1, -1>(v96 * v99, v99, v98, v65, v94);
  // pto: %rms_partials_inline825__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v102 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v100, v101);
  // pto: %x_flat_inline776__ssa_v0_view
  const int64_t v103 = 1;
  // pto: %x_flat_inline776__ssa_v0_view
  const int64_t v104 = 1;
  // pto: %x_flat_inline776__ssa_v0_view
  const int64_t v105 = 1;
  // pto: %x_flat_inline776__ssa_v0_view
  int64_t v106 = (int64_t) v30;
  // pto: %x_flat_inline776__ssa_v0_view
  int64_t v107 = v106 * v67;
  // pto: %x_flat_inline776__ssa_v0_view
  int64_t v108 = v105 * v107;
  // pto: %x_flat_inline776__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v109 = pto::Shape<1, 1, 1, -1, -1>(v103, v104, v105, v106, v67);
  // pto: %x_flat_inline776__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v110 = pto::Stride<-1, -1, -1, -1, -1>(v104 * v108, v108, v107, v67, v65);
  // pto: %x_flat_inline776__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v111 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v109, v110);
  // pto: %222
  int64_t v112 = (int64_t) ((uint64_t) v29 * (uint64_t) v64);
  // pto: %parts_2d_inline779__ssa_v0_view
  const int64_t v113 = 1;
  // pto: %parts_2d_inline779__ssa_v0_view
  const int64_t v114 = 1;
  // pto: %parts_2d_inline779__ssa_v0_view
  const int64_t v115 = 1;
  // pto: %parts_2d_inline779__ssa_v0_view
  int64_t v116 = v69 * v112;
  // pto: %parts_2d_inline779__ssa_v0_view
  int64_t v117 = v115 * v116;
  // pto: %parts_2d_inline779__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v118 = pto::Shape<1, 1, 1, -1, -1>(v113, v114, v115, v69, v112);
  // pto: %parts_2d_inline779__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v119 = pto::Stride<-1, -1, -1, -1, -1>(v114 * v117, v117, v116, v112, v65);
  // pto: %parts_2d_inline779__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v120 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v118, v119);
  // pto: %rms_2d_inline856__ssa_v0_view
  const int64_t v121 = 1;
  // pto: %rms_2d_inline856__ssa_v0_view
  const int64_t v122 = 1;
  // pto: %rms_2d_inline856__ssa_v0_view
  const int64_t v123 = 1;
  // pto: %rms_2d_inline856__ssa_v0_view
  int64_t v124 = (int64_t) v29;
  // pto: %rms_2d_inline856__ssa_v0_view
  int64_t v125 = v69 * v124;
  // pto: %rms_2d_inline856__ssa_v0_view
  int64_t v126 = v123 * v125;
  // pto: %rms_2d_inline856__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v127 = pto::Shape<1, 1, 1, -1, -1>(v121, v122, v123, v69, (int64_t) v29);
  // pto: %rms_2d_inline856__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v128 = pto::Stride<-1, -1, -1, -1, -1>(v122 * v126, v126, v125, v124, v65);
  // pto: %rms_2d_inline856__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v129 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v127, v128);
  // pto: %hc_base_2d_inline816__ssa_v0_view
  const int64_t v130 = 1;
  // pto: %hc_base_2d_inline816__ssa_v0_view
  const int64_t v131 = 1;
  // pto: %hc_base_2d_inline816__ssa_v0_view
  const int64_t v132 = 1;
  // pto: %hc_base_2d_inline816__ssa_v0_view
  int64_t v133 = v65 * v66;
  // pto: %hc_base_2d_inline816__ssa_v0_view
  int64_t v134 = v132 * v133;
  // pto: %hc_base_2d_inline816__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v135 = pto::Shape<1, 1, 1, -1, -1>(v130, v131, v132, v65, v66);
  // pto: %hc_base_2d_inline816__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v136 = pto::Stride<-1, -1, -1, -1, -1>(v131 * v134, v134, v133, v66, v65);
  // pto: %hc_base_2d_inline816__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v137 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v8, v135, v136);
  // pto: %comb_inline720__ssa_v0_view
  const int64_t v138 = 1;
  // pto: %comb_inline720__ssa_v0_view
  const int64_t v139 = 1;
  // pto: %comb_inline720__ssa_v0_view
  const int64_t v140 = 1;
  // pto: %comb_inline720__ssa_v0_view
  int64_t v141 = (int64_t) v30;
  // pto: %comb_inline720__ssa_v0_view
  int64_t v142 = v141 * v70;
  // pto: %comb_inline720__ssa_v0_view
  int64_t v143 = v140 * v142;
  // pto: %comb_inline720__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v144 = pto::Shape<1, 1, 1, -1, -1>(v138, v139, v140, v141, v70);
  // pto: %comb_inline720__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v145 = pto::Stride<-1, -1, -1, -1, -1>(v139 * v143, v143, v142, v70, v65);
  // pto: %comb_inline720__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v146 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v9, v144, v145);
  // pto: %hc_scale_2d_inline757__ssa_v0_view
  const int64_t v147 = 1;
  // pto: %hc_scale_2d_inline757__ssa_v0_view
  const int64_t v148 = 1;
  // pto: %hc_scale_2d_inline757__ssa_v0_view
  const int64_t v149 = 1;
  // pto: %hc_scale_2d_inline757__ssa_v0_view
  int64_t v150 = v65 * v68;
  // pto: %hc_scale_2d_inline757__ssa_v0_view
  int64_t v151 = v149 * v150;
  // pto: %hc_scale_2d_inline757__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v152 = pto::Shape<1, 1, 1, -1, -1>(v147, v148, v149, v65, v68);
  // pto: %hc_scale_2d_inline757__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v153 = pto::Stride<-1, -1, -1, -1, -1>(v148 * v151, v151, v150, v68, v65);
  // pto: %hc_scale_2d_inline757__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v154 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v10, v152, v153);
  // pto: %223
  int64_t v155 = (int64_t) ((uint64_t) v30 * (uint64_t) v78);
  // pto: %post_flat_inline781__ssa_v0_view
  const int64_t v156 = 1;
  // pto: %post_flat_inline781__ssa_v0_view
  const int64_t v157 = 1;
  // pto: %post_flat_inline781__ssa_v0_view
  const int64_t v158 = 1;
  // pto: %post_flat_inline781__ssa_v0_view
  int64_t v159 = v65 * v155;
  // pto: %post_flat_inline781__ssa_v0_view
  int64_t v160 = v158 * v159;
  // pto: %post_flat_inline781__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v161 = pto::Shape<1, 1, 1, -1, -1>(v156, v157, v158, v65, v155);
  // pto: %post_flat_inline781__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v162 = pto::Stride<-1, -1, -1, -1, -1>(v157 * v160, v160, v159, v155, v65);
  // pto: %post_flat_inline781__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v163 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v11, v161, v162);
  // pto: %mixed_view_inline726__ssa_v0_view
  const int64_t v164 = 1;
  // pto: %mixed_view_inline726__ssa_v0_view
  const int64_t v165 = 1;
  // pto: %mixed_view_inline726__ssa_v0_view
  const int64_t v166 = 1;
  // pto: %mixed_view_inline726__ssa_v0_view
  int64_t v167 = (int64_t) v30;
  // pto: %mixed_view_inline726__ssa_v0_view
  int64_t v168 = v167 * v71;
  // pto: %mixed_view_inline726__ssa_v0_view
  int64_t v169 = v166 * v168;
  // pto: %mixed_view_inline726__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v170 = pto::Shape<1, 1, 1, -1, -1>(v164, v165, v166, v167, v71);
  // pto: %mixed_view_inline726__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v171 = pto::Stride<-1, -1, -1, -1, -1>(v165 * v169, v169, v168, v71, v65);
  // pto: %mixed_view_inline726__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v172 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v12, v170, v171);
  const int32_t v173 = 0;
  auto v174 = TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>(v13, v173, v72);
  // pto: %worker_inline766__ssa_v0
  int64_t v175 = (int64_t) v31;
  // pto: %linear_workers_inline925__ssa_v0_idx
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  set_flag(PIPE_S, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
  for (int64_t v176 = v175; v176 < v14; v176 += (int64_t) v15) {
    // pto: %36, %37
    ;
    int64_t v177 = (int64_t) ((uint64_t) (v176 / v69) * (uint64_t) v64);
    // pto: %38
    ;
    int64_t v178 = v176 % v69;
    // pto: %mid_stream_rms_total_inline714__tile
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v179 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v70);
    // pto: %mid_stream_rms_total_inline714__tile
    ;
    uint64_t v180 = (uint64_t) v34;
    TASSIGN(v179, v180);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TEXPANDS(v179, v74);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    for (int64_t v181 = v75; v181 < v76; v181 += v65) {
      // pto: %39, %43, %42
      ;
      int64_t v182 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v178 * (uint64_t) v73) + (uint64_t) ((int64_t) (uint64_t) v181 * (uint64_t) v77));
      // pto: %45, %lane_inline945__ssa_v0, %44
      ;
      int64_t v183 = (int64_t) ((uint64_t) v177 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) v33) * (uint64_t) v70));
      // pto: %46
      ;
      int64_t v184 = (int64_t) ((uint64_t) v17 - (uint64_t) v183);
      // pto: %47
      ;
      int64_t v185 = v184 < v70 ? v184 : v70;
      // pto: %48
      ;
      int64_t v186 = v185 < v75 ? v75 : v185;
      // pto: %mid_stream_source_inline709__tile
      ;
      Tile<TileType::Vec, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v187 = Tile<TileType::Vec, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v186, v77);
      // pto: %mid_stream_source_inline709__tile
      ;
      uint64_t v188 = (uint64_t) v35;
      TASSIGN(v187, v188);
      // pto: %49
      ;
      int64_t v189 = v183 < v75 ? v75 : v183;
      // pto: %50
      ;
      int64_t v190 = v182 < v75 ? v75 : v182;
      // pto: %x_flat_inline776__ssa_v0_pview
      ;
      const int64_t v191 = 0;
      // pto: %x_flat_inline776__ssa_v0_pview
      ;
      __gm__ bfloat16_t* v192 = PTOAS__GLOBAL_TENSOR_DATA(v111);
      // pto: %x_flat_inline776__ssa_v0_pview
      ;
      const int64_t v193 = 1;
      // pto: %x_flat_inline776__ssa_v0_pview
      ;
      const int64_t v194 = 1;
      // pto: %x_flat_inline776__ssa_v0_pview
      ;
      const int64_t v195 = 1;
      // pto: %x_flat_inline776__ssa_v0_pview
      ;
      int64_t v196 = v186 * v67;
      // pto: %x_flat_inline776__ssa_v0_pview
      ;
      int64_t v197 = v195 * v196;
      // pto: %x_flat_inline776__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, -1> v198 = pto::Shape<1, 1, 1, -1, -1>(v193, v194, v195, v186, v77);
      // pto: %x_flat_inline776__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v199 = pto::Stride<-1, -1, -1, -1, -1>(v194 * v197, v197, v196, v67, v65);
      // pto: %x_flat_inline776__ssa_v0_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v200 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v192 + (v191 + v189 * v67 + v190 * v65), v198, v199);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v187, v200);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %mid_stream_widened_inline803__tile
      ;
      Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v201 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v186, v77);
      // pto: %mid_stream_widened_inline803__tile
      ;
      uint64_t v202 = (uint64_t) v36;
      TASSIGN(v201, v202);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      RoundMode v203 = RoundMode::CAST_ROUND;
      SaturationMode v204 = SaturationMode::ON;
      TCVT(v201, v187, v203, v204);
      v201.SetValidShape(v186, v77);
      // pto: %mid_stream_clean_inline732__tile
      ;
      Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v205 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v70, v77);
      // pto: %mid_stream_clean_inline732__tile
      ;
      uint64_t v206 = (uint64_t) v36;
      TASSIGN(v205, v206);
      pipe_barrier(PIPE_V);
      TFILLPAD<pto::TFillPadMode::InPlace>(v205, v201);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      v205.SetValidShape(v70, v77);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TPUSH<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v174, v205);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      // pto: %0
      ;
      Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v207 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v70, v77);
      // pto: %0
      ;
      uint64_t v208 = (uint64_t) v36;
      TASSIGN(v207, v208);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      TMUL(v207, v205, v205);
      // pto: %tmp_tile
      ;
      Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v209 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v77);
      // pto: %tmp_tile
      ;
      uint64_t v210 = (uint64_t) v35;
      TASSIGN(v209, v210);
      // pto: %1
      ;
      Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v211 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v65);
      // pto: %1
      ;
      uint64_t v212 = (uint64_t) v37;
      TASSIGN(v211, v212);
      pipe_barrier(PIPE_V);
      TROWSUM(v211, v207, v209);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      // pto: %mid_stream_sq_inline725__tile
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v213 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v70);
      // pto: %mid_stream_sq_inline725__tile
      ;
      uint64_t v214 = (uint64_t) v37;
      TASSIGN(v213, v214);
      // pto: %2
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v215 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v70);
      // pto: %2
      ;
      uint64_t v216 = (uint64_t) v34;
      TASSIGN(v215, v216);
      pipe_barrier(PIPE_V);
      TADD(v215, v179, v213);
    };
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %40, %41, %52, %rms_lane_inline724__ssa_v0, %51
    ;
    int64_t v217 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v178 * (uint64_t) v16) + (uint64_t) v177) + (uint64_t) ((int64_t) (uint64_t) ((int64_t) v33) * (uint64_t) v70));
    // pto: %3
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v218 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v65);
    // pto: %3
    ;
    uint64_t v219 = (uint64_t) v34;
    TASSIGN(v218, v219);
    // pto: %53
    ;
    int64_t v220 = v217 < v75 ? v75 : v217;
    // pto: %rms_partials_inline825__iter_v1_pview
    ;
    const int64_t v221 = 0;
    // pto: %rms_partials_inline825__iter_v1_pview
    ;
    __gm__ float* v222 = PTOAS__GLOBAL_TENSOR_DATA(v102);
    // pto: %rms_partials_inline825__iter_v1_pview
    ;
    const int64_t v223 = 1;
    // pto: %rms_partials_inline825__iter_v1_pview
    ;
    const int64_t v224 = 1;
    // pto: %rms_partials_inline825__iter_v1_pview
    ;
    const int64_t v225 = 1;
    // pto: %rms_partials_inline825__iter_v1_pview
    ;
    int64_t v226 = v70 * v65;
    // pto: %rms_partials_inline825__iter_v1_pview
    ;
    int64_t v227 = v225 * v226;
    // pto: %rms_partials_inline825__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 16, 1> v228 = pto::Shape<1, 1, 1, 16, 1>(v223, v224, v225, v70, v65);
    // pto: %rms_partials_inline825__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v229 = pto::Stride<-1, -1, -1, -1, -1>(v224 * v227, v227, v226, v65, v94);
    // pto: %rms_partials_inline825__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v230 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v222 + (v221 + v220 * v65 + v75 * v94), v228, v229);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    TSTORE(v230, v218);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  pipe_barrier(PIPE_ALL);
  dsb(DSB_DDR);
  SYNCALL<SyncCoreType::Mix>();
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  // pto: %54, %55, %tail_lane_inline700__ssa_v0
  int64_t v231 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v175 * (uint64_t) v76) + (uint64_t) ((int64_t) v33));
  // pto: %56
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  if (v231 < v18) {
    // pto: %58
    ;
    int64_t v232 = v231 < v20 ? v231 : v20;
    // pto: %57, %59
    ;
    int64_t v233 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v231 * (uint64_t) v19) + (uint64_t) v232);
    // pto: %61
    ;
    int64_t v234 = (int64_t) ((uint64_t) v20 - (uint64_t) v231);
    // pto: %62
    ;
    int64_t v235 = v234 < v75 ? v75 : v234;
    // pto: %63
    ;
    int64_t v236 = v235 < v65 ? v235 : v65;
    // pto: %60, %64
    ;
    int64_t v237 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v233 + (uint64_t) v19) + (uint64_t) v236);
    for (int64_t v238 = v233; v238 < v237; v238 += v78) {
      // pto: %scale2_inline777__tile
      ;
      float v239 = (v5)[v76];
      // pto: %65
      ;
      int64_t v240 = (int64_t) ((uint64_t) v237 - (uint64_t) v238);
      // pto: %66
      ;
      int64_t v241 = v240 < v78 ? v240 : v78;
      // pto: %67
      ;
      int64_t v242 = (int64_t) ((uint64_t) v241 * (uint64_t) v64);
      // pto: %c_parts_load_inline807__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v243 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v242);
      // pto: %c_parts_load_inline807__ssa_v0
      ;
      uint64_t v244 = (uint64_t) v36;
      TASSIGN(v243, v244);
      // pto: %68
      ;
      int64_t v245 = (int64_t) ((uint64_t) v238 * (uint64_t) v64);
      // pto: %69
      ;
      int64_t v246 = v245 < v75 ? v75 : v245;
      // pto: %parts_2d_inline779__ssa_v0_pview
      ;
      const int64_t v247 = 0;
      // pto: %parts_2d_inline779__ssa_v0_pview
      ;
      __gm__ float* v248 = PTOAS__GLOBAL_TENSOR_DATA(v120);
      // pto: %parts_2d_inline779__ssa_v0_pview
      ;
      const int64_t v249 = 1;
      // pto: %parts_2d_inline779__ssa_v0_pview
      ;
      const int64_t v250 = 1;
      // pto: %parts_2d_inline779__ssa_v0_pview
      ;
      const int64_t v251 = 1;
      // pto: %parts_2d_inline779__ssa_v0_pview
      ;
      int64_t v252 = v69 * v112;
      // pto: %parts_2d_inline779__ssa_v0_pview
      ;
      int64_t v253 = v251 * v252;
      // pto: %parts_2d_inline779__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, -1> v254 = pto::Shape<1, 1, 1, 8, -1>(v249, v250, v251, v69, v242);
      // pto: %parts_2d_inline779__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v255 = pto::Stride<-1, -1, -1, -1, -1>(v250 * v253, v253, v252, v112, v65);
      // pto: %parts_2d_inline779__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v256 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v248 + (v247 + v75 * v112 + v246 * v65), v254, v255);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      TLOAD(v243, v256);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %t__tmp_v166
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v257 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v69, v79);
      // pto: %t__tmp_v166
      ;
      uint64_t v258 = (uint64_t) v36;
      TASSIGN(v257, v258);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      TFILLPAD<pto::TFillPadMode::InPlace>(v257, v243);
      v257.SetValidShape(v69, v79);
      // pto: %c_parts_tmp_inline733__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v259 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v79);
      // pto: %c_parts_tmp_inline733__ssa_v0
      ;
      uint64_t v260 = (uint64_t) v38;
      TASSIGN(v259, v260);
      // pto: %t__tmp_v167
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v261 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v79);
      // pto: %t__tmp_v167
      ;
      uint64_t v262 = (uint64_t) v36;
      TASSIGN(v261, v262);
      // pto: %colsum_src_view
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, 8, 256, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v263;
      TRESHAPE(v263, v257);
      // pto: %colsum_tmp_view
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, 8, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v264;
      TRESHAPE(v264, v259);
      // pto: %colsum_dst_view
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v265;
      TRESHAPE(v265, v261);
      pipe_barrier(PIPE_V);
      const bool v266 = true;
      TCOLSUM(v265, v263, v264, v266);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
      // pto: %c_total_inline690__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v267 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v64);
      // pto: %c_total_inline690__ssa_v0
      ;
      uint64_t v268 = (uint64_t) v36;
      TASSIGN(v267, v268);
      // pto: %c_sq_load_inline773__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v269 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v241);
      // pto: %c_sq_load_inline773__ssa_v0
      ;
      uint64_t v270 = (uint64_t) v39;
      TASSIGN(v269, v270);
      // pto: %71
      ;
      int64_t v271 = v238 < v75 ? v75 : v238;
      // pto: %rms_2d_inline856__ssa_v0_pview
      ;
      int64_t v272 = (int64_t) v29;
      // pto: %rms_2d_inline856__ssa_v0_pview
      ;
      const int64_t v273 = 0;
      // pto: %rms_2d_inline856__ssa_v0_pview
      ;
      __gm__ float* v274 = PTOAS__GLOBAL_TENSOR_DATA(v129);
      // pto: %rms_2d_inline856__ssa_v0_pview
      ;
      const int64_t v275 = 1;
      // pto: %rms_2d_inline856__ssa_v0_pview
      ;
      const int64_t v276 = 1;
      // pto: %rms_2d_inline856__ssa_v0_pview
      ;
      const int64_t v277 = 1;
      // pto: %rms_2d_inline856__ssa_v0_pview
      ;
      int64_t v278 = v69 * v272;
      // pto: %rms_2d_inline856__ssa_v0_pview
      ;
      int64_t v279 = v277 * v278;
      // pto: %rms_2d_inline856__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, -1> v280 = pto::Shape<1, 1, 1, 8, -1>(v275, v276, v277, v69, v241);
      // pto: %rms_2d_inline856__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v281 = pto::Stride<-1, -1, -1, -1, -1>(v276 * v279, v279, v278, v272, v65);
      // pto: %rms_2d_inline856__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v282 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v274 + (v273 + v75 * v272 + v271 * v65), v280, v281);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
      TLOAD(v269, v282);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %t__tmp_v168
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v283 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v69, v69);
      // pto: %t__tmp_v168
      ;
      uint64_t v284 = (uint64_t) v39;
      TASSIGN(v283, v284);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      TFILLPAD<pto::TFillPadMode::InPlace>(v283, v269);
      v283.SetValidShape(v69, v69);
      // pto: %c_sq_tmp_inline846__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v285 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v69);
      // pto: %c_sq_tmp_inline846__ssa_v0
      ;
      uint64_t v286 = (uint64_t) v40;
      TASSIGN(v285, v286);
      // pto: %c_rms_tmp_inline940__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v287 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %c_rms_tmp_inline940__ssa_v0
      ;
      uint64_t v288 = (uint64_t) v41;
      TASSIGN(v287, v288);
      // pto: %t__tmp_v169
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v289 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v169
      ;
      uint64_t v290 = (uint64_t) v39;
      TASSIGN(v289, v290);
      // pto: %72
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v291;
      TRESHAPE(v291, v283);
      // pto: %73
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v292;
      TRESHAPE(v292, v285);
      // pto: %74
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v293;
      TRESHAPE(v293, v289);
      pipe_barrier(PIPE_V);
      const bool v294 = true;
      TCOLSUM(v293, v291, v292, v294);
      // pto: %t__tmp_v170
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v295 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v170
      ;
      uint64_t v296 = (uint64_t) v39;
      TASSIGN(v295, v296);
      pipe_barrier(PIPE_V);
      TMULS(v295, v289, v80);
      // pto: %t__tmp_v171
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v297 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v171
      ;
      uint64_t v298 = (uint64_t) v39;
      TASSIGN(v297, v298);
      pipe_barrier(PIPE_V);
      TADDS(v297, v295, v81);
      // pto: %t__tmp_v172
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v299 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v172
      ;
      uint64_t v300 = (uint64_t) v42;
      TASSIGN(v299, v300);
      pipe_barrier(PIPE_V);
      TRSQRT(v299, v297, v287);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      // pto: %c_inv_inline844__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v301 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v65);
      // pto: %c_inv_inline844__ssa_v0
      ;
      uint64_t v302 = (uint64_t) v42;
      TASSIGN(v301, v302);
      // pto: %comb_inv_inline855__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v303 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v65);
      // pto: %comb_inv_inline855__ssa_v0
      ;
      uint64_t v304 = (uint64_t) v42;
      TASSIGN(v303, v304);
      // pto: %t__tmp_v173
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v305 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v70);
      // pto: %t__tmp_v173
      ;
      uint64_t v306 = (uint64_t) v43;
      TASSIGN(v305, v306);
      // pto: %slice_view
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 16, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v307;
      // pto: %slice_view
      ;
      uint64_t v308 = (uint64_t) v43;
      TASSIGN(v307, v308);
      // pto: %raw_inline865__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v309 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v70);
      // pto: %raw_inline865__ssa_v0
      ;
      uint64_t v310 = (uint64_t) v36;
      TASSIGN(v309, v310);
      TADDS(v309, v307, v74);
      // pto: %base_inline827__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v311 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v70);
      // pto: %base_inline827__ssa_v0
      ;
      uint64_t v312 = (uint64_t) v39;
      TASSIGN(v311, v312);
      // pto: %hc_base_2d_inline816__ssa_v0_pview
      ;
      __gm__ float* v313 = PTOAS__GLOBAL_TENSOR_DATA(v137);
      // pto: %hc_base_2d_inline816__ssa_v0_pview
      ;
      unsigned v314 = 8;
      // pto: %hc_base_2d_inline816__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 16> v315 = pto::Shape<1, 1, 1, 1, 16>();
      // pto: %hc_base_2d_inline816__ssa_v0_pview
      ;
      pto::Stride<24, 24, 24, 24, 1> v316 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %hc_base_2d_inline816__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v317 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v313 + v314, v315, v316);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      TLOAD(v311, v317);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      // pto: %t__tmp_v174
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v318 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v70);
      // pto: %t__tmp_v174
      ;
      uint64_t v319 = (uint64_t) v44;
      TASSIGN(v318, v319);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v318, v309, v303);
      // pto: %t__tmp_v175
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v320 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v70);
      // pto: %t__tmp_v175
      ;
      uint64_t v321 = (uint64_t) v44;
      TASSIGN(v320, v321);
      pipe_barrier(PIPE_V);
      TMULS(v320, v318, v239);
      // pto: %t__tmp_v176
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v322 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v70);
      // pto: %t__tmp_v176
      ;
      uint64_t v323 = (uint64_t) v36;
      TASSIGN(v322, v323);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      TCOLEXPAND(v322, v311);
      // pto: %logits_inline735__ssa_v1
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v324 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v70);
      // pto: %logits_inline735__ssa_v1
      ;
      uint64_t v325 = (uint64_t) v45;
      TASSIGN(v324, v325);
      pipe_barrier(PIPE_V);
      TADD(v324, v320, v322);
      // pto: %t__tmp_v177
      ;
      Tile<TileType::Vec, float, 4, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v326 = Tile<TileType::Vec, float, 4, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v70);
      // pto: %t__tmp_v177
      ;
      uint64_t v327 = (uint64_t) v45;
      TASSIGN(v326, v327);
      // pto: %regular_logits_inline728__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v328 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %regular_logits_inline728__ssa_v0
      ;
      uint64_t v329 = (uint64_t) v45;
      TASSIGN(v328, v329);
      // pto: %ids_inline859__ci_tmp_v0
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v330 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v83);
      // pto: %ids_inline859__ci_tmp_v0
      ;
      uint64_t v331 = (uint64_t) v36;
      TASSIGN(v330, v331);
      // pto: %ids_inline859__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v332 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %ids_inline859__ssa_v1
      ;
      uint64_t v333 = (uint64_t) v40;
      TASSIGN(v332, v333);
      // pto: %ci_tmp_view
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v334;
      TRESHAPE(v334, v330);
      // pto: %ci_dst_view
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v335;
      TRESHAPE(v335, v332);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v335, v72, v334);
      // pto: %tmp_inline793__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v336 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %tmp_inline793__ssa_v1
      ;
      uint64_t v337 = (uint64_t) v36;
      TASSIGN(v336, v337);
      // pto: %ids16_inline699__ssa_v1
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v338 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v84);
      // pto: %ids16_inline699__ssa_v1
      ;
      uint64_t v339 = (uint64_t) v40;
      TASSIGN(v338, v339);
      // pto: %xor_tmp16_inline836__ssa_v1
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v340 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v84);
      // pto: %xor_tmp16_inline836__ssa_v1
      ;
      uint64_t v341 = (uint64_t) v46;
      TASSIGN(v340, v341);
      // pto: %t__tmp_v178
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v342 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v178
      ;
      uint64_t v343 = (uint64_t) v44;
      TASSIGN(v342, v343);
      pipe_barrier(PIPE_V);
      TEXPANDS(v342, v85);
      // pto: %ri1_mask_inline901__ssa_v1
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v344 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v84);
      // pto: %ri1_mask_inline901__ssa_v1
      ;
      uint64_t v345 = (uint64_t) v44;
      TASSIGN(v344, v345);
      // pto: %t__tmp_v179
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v346 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v84);
      // pto: %t__tmp_v179
      ;
      uint64_t v347 = (uint64_t) v41;
      TASSIGN(v346, v347);
      pipe_barrier(PIPE_V);
      TXOR(v346, v338, v344, v340);
      // pto: %ri1_inline873__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v348 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %ri1_inline873__ssa_v1
      ;
      uint64_t v349 = (uint64_t) v41;
      TASSIGN(v348, v349);
      // pto: %t__tmp_v180
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v350 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v180
      ;
      uint64_t v351 = (uint64_t) v44;
      TASSIGN(v350, v351);
      pipe_barrier(PIPE_V);
      TEXPANDS(v350, v86);
      // pto: %ri2_mask_inline897__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v352 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v84);
      // pto: %ri2_mask_inline897__ssa_v0
      ;
      uint64_t v353 = (uint64_t) v44;
      TASSIGN(v352, v353);
      // pto: %t__tmp_v181
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v354 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v84);
      // pto: %t__tmp_v181
      ;
      uint64_t v355 = (uint64_t) v47;
      TASSIGN(v354, v355);
      pipe_barrier(PIPE_V);
      TXOR(v354, v338, v352, v340);
      // pto: %ri2_inline813__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v356 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %ri2_inline813__ssa_v0
      ;
      uint64_t v357 = (uint64_t) v47;
      TASSIGN(v356, v357);
      // pto: %t__tmp_v182
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v358 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v182
      ;
      uint64_t v359 = (uint64_t) v44;
      TASSIGN(v358, v359);
      pipe_barrier(PIPE_V);
      TEXPANDS(v358, v87);
      // pto: %ci1_mask_inline900__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v360 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v84);
      // pto: %ci1_mask_inline900__ssa_v0
      ;
      uint64_t v361 = (uint64_t) v44;
      TASSIGN(v360, v361);
      // pto: %t__tmp_v183
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v362 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v84);
      // pto: %t__tmp_v183
      ;
      uint64_t v363 = (uint64_t) v48;
      TASSIGN(v362, v363);
      pipe_barrier(PIPE_V);
      TXOR(v362, v338, v360, v340);
      // pto: %ci1_inline842__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v364 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %ci1_inline842__ssa_v0
      ;
      uint64_t v365 = (uint64_t) v48;
      TASSIGN(v364, v365);
      // pto: %t__tmp_v184
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v366 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v184
      ;
      uint64_t v367 = (uint64_t) v44;
      TASSIGN(v366, v367);
      pipe_barrier(PIPE_V);
      TEXPANDS(v366, v88);
      // pto: %ci2_mask_inline905__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v368 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v84);
      // pto: %ci2_mask_inline905__ssa_v0
      ;
      uint64_t v369 = (uint64_t) v44;
      TASSIGN(v368, v369);
      // pto: %t__tmp_v185
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v370 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v84);
      // pto: %t__tmp_v185
      ;
      uint64_t v371 = (uint64_t) v49;
      TASSIGN(v370, v371);
      pipe_barrier(PIPE_V);
      TXOR(v370, v338, v368, v340);
      // pto: %ci2_inline907__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v372 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %ci2_inline907__ssa_v0
      ;
      uint64_t v373 = (uint64_t) v49;
      TASSIGN(v372, v373);
      // pto: %t__tmp_v186
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v374 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v186
      ;
      uint64_t v375 = (uint64_t) v46;
      TASSIGN(v374, v375);
      pipe_barrier(PIPE_V);
      TGATHER(v374, v328, v348, v336);
      // pto: %max2_inline880__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v376 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %max2_inline880__ssa_v1
      ;
      uint64_t v377 = (uint64_t) v46;
      TASSIGN(v376, v377);
      pipe_barrier(PIPE_V);
      TMAX(v376, v328, v374);
      // pto: %t__tmp_v187
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v378 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v187
      ;
      uint64_t v379 = (uint64_t) v44;
      TASSIGN(v378, v379);
      pipe_barrier(PIPE_V);
      TGATHER(v378, v376, v356, v336);
      // pto: %max4_inline814__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v380 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %max4_inline814__ssa_v1
      ;
      uint64_t v381 = (uint64_t) v46;
      TASSIGN(v380, v381);
      pipe_barrier(PIPE_V);
      TMAX(v380, v376, v378);
      // pto: %t__tmp_v188
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v382 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v188
      ;
      uint64_t v383 = (uint64_t) v46;
      TASSIGN(v382, v383);
      pipe_barrier(PIPE_V);
      TSUB(v382, v328, v380);
      // pto: %exps_inline885__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v384 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %exps_inline885__ssa_v1
      ;
      uint64_t v385 = (uint64_t) v46;
      TASSIGN(v384, v385);
      pipe_barrier(PIPE_V);
      TEXP(v384, v382);
      // pto: %t__tmp_v189
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v386 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v189
      ;
      uint64_t v387 = (uint64_t) v44;
      TASSIGN(v386, v387);
      pipe_barrier(PIPE_V);
      TGATHER(v386, v384, v348, v336);
      // pto: %sum2_inline884__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v388 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %sum2_inline884__ssa_v1
      ;
      uint64_t v389 = (uint64_t) v44;
      TASSIGN(v388, v389);
      pipe_barrier(PIPE_V);
      TADD(v388, v384, v386);
      // pto: %t__tmp_v190
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v390 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v190
      ;
      uint64_t v391 = (uint64_t) v45;
      TASSIGN(v390, v391);
      pipe_barrier(PIPE_V);
      TGATHER(v390, v388, v356, v336);
      // pto: %sum4_inline750__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v392 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %sum4_inline750__ssa_v1
      ;
      uint64_t v393 = (uint64_t) v44;
      TASSIGN(v392, v393);
      pipe_barrier(PIPE_V);
      TADD(v392, v388, v390);
      // pto: %t__tmp_v191
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v394 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v191
      ;
      uint64_t v395 = (uint64_t) v46;
      TASSIGN(v394, v395);
      pipe_barrier(PIPE_V);
      TDIV(v394, v384, v392);
      // pto: %probs_inline887__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v396 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %probs_inline887__ssa_v1
      ;
      uint64_t v397 = (uint64_t) v46;
      TASSIGN(v396, v397);
      pipe_barrier(PIPE_V);
      TADDS(v396, v394, v81);
      // pto: %t__tmp_v192
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v398 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v192
      ;
      uint64_t v399 = (uint64_t) v44;
      TASSIGN(v398, v399);
      pipe_barrier(PIPE_V);
      TGATHER(v398, v396, v364, v336);
      // pto: %col2_inline930__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v400 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %col2_inline930__ssa_v4
      ;
      uint64_t v401 = (uint64_t) v44;
      TASSIGN(v400, v401);
      pipe_barrier(PIPE_V);
      TADD(v400, v396, v398);
      // pto: %t__tmp_v193
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v402 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v193
      ;
      uint64_t v403 = (uint64_t) v45;
      TASSIGN(v402, v403);
      pipe_barrier(PIPE_V);
      TGATHER(v402, v400, v372, v336);
      // pto: %t__tmp_v194
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v404 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v194
      ;
      uint64_t v405 = (uint64_t) v44;
      TASSIGN(v404, v405);
      pipe_barrier(PIPE_V);
      TADD(v404, v400, v402);
      // pto: %col4_inline845__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v406 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %col4_inline845__ssa_v4
      ;
      uint64_t v407 = (uint64_t) v44;
      TASSIGN(v406, v407);
      pipe_barrier(PIPE_V);
      TADDS(v406, v404, v81);
      // pto: %cur_inline800__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v408 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %cur_inline800__ssa_v4
      ;
      uint64_t v409 = (uint64_t) v46;
      TASSIGN(v408, v409);
      pipe_barrier(PIPE_V);
      TDIV(v408, v396, v406);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      for (int64_t v410 = v75; v410 < v89; v410 += v65) {
        // pto: %t__tmp_v195
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v411 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %t__tmp_v195
        ;
        uint64_t v412 = (uint64_t) v44;
        TASSIGN(v411, v412);
        pipe_barrier(PIPE_V);
        TGATHER(v411, v408, v348, v336);
        // pto: %row2_inline904__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v413 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %row2_inline904__ssa_v1
        ;
        uint64_t v414 = (uint64_t) v44;
        TASSIGN(v413, v414);
        pipe_barrier(PIPE_V);
        TADD(v413, v408, v411);
        // pto: %t__tmp_v196
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v415 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %t__tmp_v196
        ;
        uint64_t v416 = (uint64_t) v45;
        TASSIGN(v415, v416);
        pipe_barrier(PIPE_V);
        TGATHER(v415, v413, v356, v336);
        // pto: %t__tmp_v197
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v417 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %t__tmp_v197
        ;
        uint64_t v418 = (uint64_t) v44;
        TASSIGN(v417, v418);
        pipe_barrier(PIPE_V);
        TADD(v417, v413, v415);
        // pto: %row4_inline740__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v419 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %row4_inline740__ssa_v1
        ;
        uint64_t v420 = (uint64_t) v44;
        TASSIGN(v419, v420);
        pipe_barrier(PIPE_V);
        TADDS(v419, v417, v81);
        // pto: %norm_inline894__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v421 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %norm_inline894__ssa_v1
        ;
        uint64_t v422 = (uint64_t) v44;
        TASSIGN(v421, v422);
        pipe_barrier(PIPE_V);
        TDIV(v421, v408, v419);
        // pto: %t__tmp_v198
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v423 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %t__tmp_v198
        ;
        uint64_t v424 = (uint64_t) v45;
        TASSIGN(v423, v424);
        pipe_barrier(PIPE_V);
        TGATHER(v423, v421, v364, v336);
        // pto: %col2_inline930__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v425 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %col2_inline930__ssa_v7
        ;
        uint64_t v426 = (uint64_t) v45;
        TASSIGN(v425, v426);
        pipe_barrier(PIPE_V);
        TADD(v425, v421, v423);
        // pto: %t__tmp_v199
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v427 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %t__tmp_v199
        ;
        uint64_t v428 = (uint64_t) v39;
        TASSIGN(v427, v428);
        pipe_barrier(PIPE_V);
        TGATHER(v427, v425, v372, v336);
        // pto: %t__tmp_v200
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v429 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %t__tmp_v200
        ;
        uint64_t v430 = (uint64_t) v45;
        TASSIGN(v429, v430);
        pipe_barrier(PIPE_V);
        TADD(v429, v425, v427);
        // pto: %col4_inline845__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v431 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %col4_inline845__ssa_v7
        ;
        uint64_t v432 = (uint64_t) v45;
        TASSIGN(v431, v432);
        pipe_barrier(PIPE_V);
        TADDS(v431, v429, v81);
        // pto: %cur_inline800__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v433 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %cur_inline800__ssa_v7
        ;
        uint64_t v434 = (uint64_t) v46;
        TASSIGN(v433, v434);
        pipe_barrier(PIPE_V);
        TDIV(v433, v421, v431);
      };
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      // pto: %t__tmp_v201
      ;
      Tile<TileType::Vec, float, 4, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v435 = Tile<TileType::Vec, float, 4, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v70);
      // pto: %t__tmp_v201
      ;
      uint64_t v436 = (uint64_t) v46;
      TASSIGN(v435, v436);
      v435.SetValidShape(v241, v70);
      // pto: %comb_inline720__ssa_v0_pview
      ;
      const int64_t v437 = 0;
      // pto: %comb_inline720__ssa_v0_pview
      ;
      __gm__ float* v438 = PTOAS__GLOBAL_TENSOR_DATA(v146);
      // pto: %comb_inline720__ssa_v0_pview
      ;
      const int64_t v439 = 1;
      // pto: %comb_inline720__ssa_v0_pview
      ;
      const int64_t v440 = 1;
      // pto: %comb_inline720__ssa_v0_pview
      ;
      const int64_t v441 = 1;
      // pto: %comb_inline720__ssa_v0_pview
      ;
      int64_t v442 = v241 * v70;
      // pto: %comb_inline720__ssa_v0_pview
      ;
      int64_t v443 = v441 * v442;
      // pto: %comb_inline720__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, 16> v444 = pto::Shape<1, 1, 1, -1, 16>(v439, v440, v441, v241, v70);
      // pto: %comb_inline720__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v445 = pto::Stride<-1, -1, -1, -1, -1>(v440 * v443, v443, v442, v70, v65);
      // pto: %comb_inline720__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v446 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v438 + (v437 + v271 * v70 + v75 * v65), v444, v445);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      TSTORE(v446, v435);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    };
  } else {
    // pto: %77
    ;
    int64_t v447 = (int64_t) ((uint64_t) v231 - (uint64_t) v18);
    // pto: %79
    ;
    int64_t v448 = v447 < v22 ? v447 : v22;
    // pto: %78, %80, %81
    ;
    int64_t v449 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v447 * (uint64_t) v21) + (uint64_t) v448) * (uint64_t) v76);
    // pto: %82
    ;
    int64_t v450 = (int64_t) ((uint64_t) v22 - (uint64_t) v447);
    // pto: %83
    ;
    int64_t v451 = v450 < v75 ? v75 : v450;
    // pto: %84
    ;
    int64_t v452 = v451 < v65 ? v451 : v65;
    // pto: %87, %85, %86
    ;
    int64_t v453 = (int64_t) ((uint64_t) v449 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v21 + (uint64_t) v452) * (uint64_t) v76));
    // pto: %88
    ;
    // pto: %89
    ;
    // pto: %d_begin_inline704__phi_v2
    ;
    int64_t v454;
    // pto: %work_begin_inline692__phi_v3
    ;
    int64_t v455;
    // pto: %work_end_inline691__phi_v3
    ;
    int64_t v456;
    if (v24 < v25) {
      // pto: %90
      ;
      // pto: %d_worker_inline716__phi_v2
      ;
      int64_t v457;
      // pto: %d_workers_inline954__phi_v2
      ;
      int64_t v458;
      // pto: %pair_id_inline829__phi_v2
      ;
      int64_t v459;
      if (v447 < v26) {
        // pto: %91
        ;
        int64_t v460 = (int64_t) ((uint64_t) v27 + (uint64_t) v65);
        // pto: %92
        ;
        // pto: %94
        ;
        v457 = v447 % v460;
        v458 = v460;
        v459 = v447 / v460;
      } else {
        // pto: %96
        ;
        int64_t v461 = (int64_t) ((uint64_t) v447 - (uint64_t) v26);
        // pto: %98, %97
        ;
        // pto: %99
        ;
        v457 = v461 % v27;
        v458 = v27;
        v459 = (int64_t) ((uint64_t) v28 + (uint64_t) (v461 / v27));
      };
      // pto: %100
      ;
      uint64_t v462 = (uint64_t) v459;
      // pto: %100
      ;
      int64_t v463 = (int64_t) (v462 * (uint64_t) v76);
      // pto: %101
      ;
      int64_t v464 = v23 < v463 ? v23 : v463;
      // pto: %102
      ;
      int64_t v465 = (int64_t) ((uint64_t) v464 + (uint64_t) v76);
      // pto: %103
      ;
      // pto: %104
      ;
      int64_t v466 = v69 / v458;
      // pto: %105
      ;
      int64_t v467 = v69 % v458;
      // pto: %106
      ;
      uint64_t v468 = (uint64_t) v457;
      // pto: %107
      ;
      bool v469 = v457 < v467;
      // pto: %107
      ;
      int64_t v470 = v469 ? v457 : v467;
      // pto: %106, %108
      ;
      v454 = (int64_t) ((uint64_t) ((int64_t) v468 * (uint64_t) v466) + (uint64_t) v470);
      v455 = v464;
      v456 = v23 < v465 ? v23 : v465;
    } else {
      v454 = v75;
      v455 = v449;
      v456 = v23 < v453 ? v23 : v453;
    };
    // pto: %109
    ;
    bool v471 = v25 <= v447;
    // pto: %work_begin_inline692__phi_v5
    ;
    int64_t v472 = v471 ? v75 : v455;
    // pto: %work_end_inline691__phi_v5
    ;
    int64_t v473 = v471 ? v75 : v456;
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    for (int64_t v474 = v472; v474 < v473; v474 += v78) {
      // pto: %110
      ;
      int64_t v475 = (int64_t) ((uint64_t) v473 - (uint64_t) v474);
      // pto: %111
      ;
      int64_t v476 = v475 < v78 ? v475 : v78;
      // pto: %112
      ;
      int64_t v477 = (int64_t) ((uint64_t) v476 * (uint64_t) v64);
      // pto: %m_parts_load_inline731__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v478 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v477);
      // pto: %m_parts_load_inline731__ssa_v0
      ;
      uint64_t v479 = (uint64_t) v36;
      TASSIGN(v478, v479);
      // pto: %113
      ;
      int64_t v480 = (int64_t) ((uint64_t) v474 * (uint64_t) v64);
      // pto: %114
      ;
      int64_t v481 = v480 < v75 ? v75 : v480;
      // pto: %116
      ;
      const int64_t v482 = 0;
      // pto: %116
      ;
      __gm__ float* v483 = PTOAS__GLOBAL_TENSOR_DATA(v120);
      // pto: %116
      ;
      const int64_t v484 = 1;
      // pto: %116
      ;
      const int64_t v485 = 1;
      // pto: %116
      ;
      const int64_t v486 = 1;
      // pto: %116
      ;
      int64_t v487 = v69 * v112;
      // pto: %116
      ;
      int64_t v488 = v486 * v487;
      // pto: %116
      ;
      pto::Shape<1, 1, 1, 8, -1> v489 = pto::Shape<1, 1, 1, 8, -1>(v484, v485, v486, v69, v477);
      // pto: %116
      ;
      pto::Stride<-1, -1, -1, -1, -1> v490 = pto::Stride<-1, -1, -1, -1, -1>(v485 * v488, v488, v487, v112, v65);
      // pto: %116
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v491 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v483 + (v482 + v75 * v112 + v481 * v65), v489, v490);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      TLOAD(v478, v491);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      // pto: %t__tmp_v202
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v492 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v69, v79);
      // pto: %t__tmp_v202
      ;
      uint64_t v493 = (uint64_t) v36;
      TASSIGN(v492, v493);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      TFILLPAD<pto::TFillPadMode::InPlace>(v492, v478);
      v492.SetValidShape(v69, v79);
      // pto: %m_parts_tmp_inline780__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v494 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v79);
      // pto: %m_parts_tmp_inline780__ssa_v0
      ;
      uint64_t v495 = (uint64_t) v38;
      TASSIGN(v494, v495);
      // pto: %t__tmp_v203
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v496 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v79);
      // pto: %t__tmp_v203
      ;
      uint64_t v497 = (uint64_t) v36;
      TASSIGN(v496, v497);
      // pto: %117
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, 8, 256, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v498;
      TRESHAPE(v498, v492);
      // pto: %118
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, 8, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v499;
      TRESHAPE(v499, v494);
      // pto: %119
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v500;
      TRESHAPE(v500, v496);
      pipe_barrier(PIPE_V);
      const bool v501 = true;
      TCOLSUM(v500, v498, v499, v501);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      // pto: %m_total_inline903__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v502 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v64);
      // pto: %m_total_inline903__ssa_v0
      ;
      uint64_t v503 = (uint64_t) v36;
      TASSIGN(v502, v503);
      // pto: %m_sq_load_inline958__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v504 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v476);
      // pto: %m_sq_load_inline958__ssa_v0
      ;
      uint64_t v505 = (uint64_t) v39;
      TASSIGN(v504, v505);
      // pto: %120
      ;
      int64_t v506 = v474 < v75 ? v75 : v474;
      // pto: %121
      ;
      int64_t v507 = (int64_t) v29;
      // pto: %121
      ;
      const int64_t v508 = 0;
      // pto: %121
      ;
      __gm__ float* v509 = PTOAS__GLOBAL_TENSOR_DATA(v129);
      // pto: %121
      ;
      const int64_t v510 = 1;
      // pto: %121
      ;
      const int64_t v511 = 1;
      // pto: %121
      ;
      const int64_t v512 = 1;
      // pto: %121
      ;
      int64_t v513 = v69 * v507;
      // pto: %121
      ;
      int64_t v514 = v512 * v513;
      // pto: %121
      ;
      pto::Shape<1, 1, 1, 8, -1> v515 = pto::Shape<1, 1, 1, 8, -1>(v510, v511, v512, v69, v476);
      // pto: %121
      ;
      pto::Stride<-1, -1, -1, -1, -1> v516 = pto::Stride<-1, -1, -1, -1, -1>(v511 * v514, v514, v513, v507, v65);
      // pto: %121
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v517 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v509 + (v508 + v75 * v507 + v506 * v65), v515, v516);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      TLOAD(v504, v517);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      // pto: %t__tmp_v204
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v518 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v69, v69);
      // pto: %t__tmp_v204
      ;
      uint64_t v519 = (uint64_t) v39;
      TASSIGN(v518, v519);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      TFILLPAD<pto::TFillPadMode::InPlace>(v518, v504);
      v518.SetValidShape(v69, v69);
      // pto: %m_sq_tmp_inline772__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v520 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v69);
      // pto: %m_sq_tmp_inline772__ssa_v0
      ;
      uint64_t v521 = (uint64_t) v40;
      TASSIGN(v520, v521);
      // pto: %m_rms_tmp_inline928__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v522 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %m_rms_tmp_inline928__ssa_v0
      ;
      uint64_t v523 = (uint64_t) v41;
      TASSIGN(v522, v523);
      // pto: %t__tmp_v205
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v524 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v205
      ;
      uint64_t v525 = (uint64_t) v39;
      TASSIGN(v524, v525);
      // pto: %122
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v526;
      TRESHAPE(v526, v518);
      // pto: %123
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v527;
      TRESHAPE(v527, v520);
      // pto: %124
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v528;
      TRESHAPE(v528, v524);
      pipe_barrier(PIPE_V);
      const bool v529 = true;
      TCOLSUM(v528, v526, v527, v529);
      // pto: %t__tmp_v206
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v530 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v206
      ;
      uint64_t v531 = (uint64_t) v39;
      TASSIGN(v530, v531);
      pipe_barrier(PIPE_V);
      TMULS(v530, v524, v80);
      // pto: %t__tmp_v207
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v532 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v207
      ;
      uint64_t v533 = (uint64_t) v39;
      TASSIGN(v532, v533);
      pipe_barrier(PIPE_V);
      TADDS(v532, v530, v81);
      // pto: %t__tmp_v208
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v534 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v208
      ;
      uint64_t v535 = (uint64_t) v47;
      TASSIGN(v534, v535);
      pipe_barrier(PIPE_V);
      TRSQRT(v534, v532, v522);
      // pto: %m_inv_inline832__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v536 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v65);
      // pto: %m_inv_inline832__ssa_v0
      ;
      uint64_t v537 = (uint64_t) v47;
      TASSIGN(v536, v537);
      // pto: %gate_base_inline959__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v538 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %gate_base_inline959__ssa_v0
      ;
      uint64_t v539 = (uint64_t) v50;
      TASSIGN(v538, v539);
      // pto: %125
      ;
      __gm__ float* v540 = PTOAS__GLOBAL_TENSOR_DATA(v137);
      // pto: %125
      ;
      pto::Shape<1, 1, 1, 1, 8> v541 = pto::Shape<1, 1, 1, 1, 8>();
      // pto: %125
      ;
      pto::Stride<24, 24, 24, 24, 1> v542 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %125
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v543 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v540, v541, v542);
      TLOAD(v538, v543);
      // pto: %gate_scales_inline774__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v544 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v68);
      // pto: %gate_scales_inline774__ssa_v0
      ;
      uint64_t v545 = (uint64_t) v51;
      TASSIGN(v544, v545);
      // pto: %hc_scale_2d_inline757__ssa_v0_pview
      ;
      __gm__ float* v546 = PTOAS__GLOBAL_TENSOR_DATA(v154);
      // pto: %hc_scale_2d_inline757__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 3> v547 = pto::Shape<1, 1, 1, 1, 3>();
      // pto: %hc_scale_2d_inline757__ssa_v0_pview
      ;
      pto::Stride<3, 3, 3, 3, 1> v548 = pto::Stride<3, 3, 3, 3, 1>();
      // pto: %hc_scale_2d_inline757__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND> v549 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND>(v546, v547, v548);
      TLOAD(v544, v549);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      // pto: %t__ci_tmp_v1
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v550 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v83);
      // pto: %t__ci_tmp_v1
      ;
      uint64_t v551 = (uint64_t) v39;
      TASSIGN(v550, v551);
      // pto: %t__tmp_v209
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v552 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v209
      ;
      uint64_t v553 = (uint64_t) v52;
      TASSIGN(v552, v553);
      // pto: %126
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v554;
      TRESHAPE(v554, v550);
      // pto: %127
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v555;
      TRESHAPE(v555, v552);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v555, v72, v554);
      // pto: %gate_ids_inline809__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v556 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %gate_ids_inline809__ssa_v0
      ;
      uint64_t v557 = (uint64_t) v39;
      TASSIGN(v556, v557);
      pipe_barrier(PIPE_V);
      RoundMode v558 = RoundMode::CAST_ROUND;
      SaturationMode v559 = SaturationMode::ON;
      TCVT(v556, v552, v558, v559);
      // pto: %t__tmp_v210
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v560 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v210
      ;
      uint64_t v561 = (uint64_t) v39;
      TASSIGN(v560, v561);
      pipe_barrier(PIPE_V);
      TMULS(v560, v556, v90);
      // pto: %gate_scale_ids_inline961__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v562 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %gate_scale_ids_inline961__ssa_v0
      ;
      uint64_t v563 = (uint64_t) v39;
      TASSIGN(v562, v563);
      pipe_barrier(PIPE_V);
      RoundMode v564 = RoundMode::CAST_FLOOR;
      SaturationMode v565 = SaturationMode::ON;
      TCVT(v562, v560, v564, v565);
      // pto: %gate_tmp_inline854__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v566 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %gate_tmp_inline854__ssa_v0
      ;
      uint64_t v567 = (uint64_t) v53;
      TASSIGN(v566, v567);
      // pto: %gate_scale_vec_inline933__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v568 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %gate_scale_vec_inline933__ssa_v0
      ;
      uint64_t v569 = (uint64_t) v42;
      TASSIGN(v568, v569);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      TGATHER(v568, v544, v562, v566);
      // pto: %t__tmp_v211
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v570 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v69);
      // pto: %t__tmp_v211
      ;
      uint64_t v571 = (uint64_t) v36;
      TASSIGN(v570, v571);
      // pto: %128
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v572;
      // pto: %128
      ;
      uint64_t v573 = (uint64_t) v36;
      TASSIGN(v572, v573);
      // pto: %gate_raw_inline899__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v574 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v69);
      // pto: %gate_raw_inline899__ssa_v0
      ;
      uint64_t v575 = (uint64_t) v36;
      TASSIGN(v574, v575);
      TADDS(v574, v572, v74);
      // pto: %gate_norm_inline963__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v576 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v69);
      // pto: %gate_norm_inline963__ssa_v0
      ;
      uint64_t v577 = (uint64_t) v36;
      TASSIGN(v576, v577);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v576, v574, v536);
      // pto: %gate_scaled_inline964__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v578 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v69);
      // pto: %gate_scaled_inline964__ssa_v0
      ;
      uint64_t v579 = (uint64_t) v36;
      TASSIGN(v578, v579);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v578, v576, v568);
      // pto: %t__tmp_v212
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v580 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v69);
      // pto: %t__tmp_v212
      ;
      uint64_t v581 = (uint64_t) v46;
      TASSIGN(v580, v581);
      TCOLEXPAND(v580, v538);
      // pto: %gate_logits_inline965__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v582 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v69);
      // pto: %gate_logits_inline965__ssa_v0
      ;
      uint64_t v583 = (uint64_t) v36;
      TASSIGN(v582, v583);
      pipe_barrier(PIPE_V);
      TADD(v582, v578, v580);
      // pto: %t__tmp_v213
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v584 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v69);
      // pto: %t__tmp_v213
      ;
      uint64_t v585 = (uint64_t) v36;
      TASSIGN(v584, v585);
      pipe_barrier(PIPE_V);
      TNEG(v584, v582);
      // pto: %t__tmp_v214
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v586 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v69);
      // pto: %t__tmp_v214
      ;
      uint64_t v587 = (uint64_t) v36;
      TASSIGN(v586, v587);
      pipe_barrier(PIPE_V);
      TEXP(v586, v584);
      // pto: %t__tmp_v215
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v588 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v69);
      // pto: %t__tmp_v215
      ;
      uint64_t v589 = (uint64_t) v36;
      TASSIGN(v588, v589);
      pipe_barrier(PIPE_V);
      TADDS(v588, v586, v91);
      // pto: %gate_sigmoid_inline689__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v590 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v69);
      // pto: %gate_sigmoid_inline689__ssa_v0
      ;
      uint64_t v591 = (uint64_t) v54;
      TASSIGN(v590, v591);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
      TRECIP(v590, v588);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      // pto: %t__ci_tmp_v2
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v592 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v83);
      // pto: %t__ci_tmp_v2
      ;
      uint64_t v593 = (uint64_t) v36;
      TASSIGN(v592, v593);
      // pto: %t__tmp_v216
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v594 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v216
      ;
      uint64_t v595 = (uint64_t) v45;
      TASSIGN(v594, v595);
      // pto: %129
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v596;
      TRESHAPE(v596, v592);
      // pto: %130
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v597;
      TRESHAPE(v597, v594);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v597, v72, v596);
      // pto: %post_ids_float_inline921__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v598 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %post_ids_float_inline921__ssa_v0
      ;
      uint64_t v599 = (uint64_t) v36;
      TASSIGN(v598, v599);
      pipe_barrier(PIPE_V);
      RoundMode v600 = RoundMode::CAST_ROUND;
      SaturationMode v601 = SaturationMode::ON;
      TCVT(v598, v594, v600, v601);
      // pto: %t__tmp_v217
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v602 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v217
      ;
      uint64_t v603 = (uint64_t) v55;
      TASSIGN(v602, v603);
      pipe_barrier(PIPE_V);
      TMULS(v602, v598, v90);
      // pto: %t__tmp_v218
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v604 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v218
      ;
      uint64_t v605 = (uint64_t) v55;
      TASSIGN(v604, v605);
      pipe_barrier(PIPE_V);
      RoundMode v606 = RoundMode::CAST_FLOOR;
      SaturationMode v607 = SaturationMode::ON;
      TCVT(v604, v602, v606, v607);
      // pto: %post_rows_inline687__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v608 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %post_rows_inline687__ssa_v0
      ;
      uint64_t v609 = (uint64_t) v55;
      TASSIGN(v608, v609);
      pipe_barrier(PIPE_V);
      RoundMode v610 = RoundMode::CAST_ROUND;
      SaturationMode v611 = SaturationMode::ON;
      TCVT(v608, v604, v610, v611);
      // pto: %t__tmp_v219
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v612 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v219
      ;
      uint64_t v613 = (uint64_t) v55;
      TASSIGN(v612, v613);
      pipe_barrier(PIPE_V);
      TMULS(v612, v608, v92);
      // pto: %t__tmp_v220
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v614 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v220
      ;
      uint64_t v615 = (uint64_t) v36;
      TASSIGN(v614, v615);
      pipe_barrier(PIPE_V);
      TADD(v614, v598, v612);
      // pto: %t__tmp_v221
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v616 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v221
      ;
      uint64_t v617 = (uint64_t) v36;
      TASSIGN(v616, v617);
      pipe_barrier(PIPE_V);
      TADDS(v616, v614, v92);
      // pto: %post_ids_inline686__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v618 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %post_ids_inline686__ssa_v0
      ;
      uint64_t v619 = (uint64_t) v36;
      TASSIGN(v618, v619);
      pipe_barrier(PIPE_V);
      RoundMode v620 = RoundMode::CAST_ROUND;
      SaturationMode v621 = SaturationMode::ON;
      TCVT(v618, v616, v620, v621);
      // pto: %post_tmp_inline685__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v622 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %post_tmp_inline685__ssa_v0
      ;
      uint64_t v623 = (uint64_t) v55;
      TASSIGN(v622, v623);
      // pto: %t__tmp_v222
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v624 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v222
      ;
      uint64_t v625 = (uint64_t) v54;
      TASSIGN(v624, v625);
      // pto: %post_values_inline742__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v626 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %post_values_inline742__ssa_v0
      ;
      uint64_t v627 = (uint64_t) v46;
      TASSIGN(v626, v627);
      pipe_barrier(PIPE_V);
      TGATHER(v626, v624, v618, v622);
      // pto: %t__tmp_v223
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v628 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v223
      ;
      uint64_t v629 = (uint64_t) v36;
      TASSIGN(v628, v629);
      pipe_barrier(PIPE_V);
      TMULS(v628, v626, v93);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      // pto: %131
      ;
      int64_t v630 = (int64_t) ((uint64_t) v476 * (uint64_t) v78);
      v628.SetValidShape(v65, v630);
      // pto: %132
      ;
      bool v631 = v454 == v75;
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      if (v631) {
        // pto: %134
        ;
        int64_t v632 = (int64_t) ((uint64_t) v474 * (uint64_t) v78);
        // pto: %135
        ;
        int64_t v633 = v632 < v75 ? v75 : v632;
        // pto: %post_flat_inline781__ssa_v0_pview
        ;
        const int64_t v634 = 0;
        // pto: %post_flat_inline781__ssa_v0_pview
        ;
        __gm__ float* v635 = PTOAS__GLOBAL_TENSOR_DATA(v163);
        // pto: %post_flat_inline781__ssa_v0_pview
        ;
        const int64_t v636 = 1;
        // pto: %post_flat_inline781__ssa_v0_pview
        ;
        const int64_t v637 = 1;
        // pto: %post_flat_inline781__ssa_v0_pview
        ;
        const int64_t v638 = 1;
        // pto: %post_flat_inline781__ssa_v0_pview
        ;
        int64_t v639 = v65 * v155;
        // pto: %post_flat_inline781__ssa_v0_pview
        ;
        int64_t v640 = v638 * v639;
        // pto: %post_flat_inline781__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, -1> v641 = pto::Shape<1, 1, 1, 1, -1>(v636, v637, v638, v65, v630);
        // pto: %post_flat_inline781__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v642 = pto::Stride<-1, -1, -1, -1, -1>(v637 * v640, v640, v639, v155, v65);
        // pto: %post_flat_inline781__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v643 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v635 + (v634 + v75 * v155 + v633 * v65), v641, v642);
        TSTORE(v643, v628);
      };
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      // pto: %136
      ;
      int64_t v644 = v476 / v76;
      // pto: %137
      ;
      int64_t v645 = (int64_t) ((uint64_t) v644 * (uint64_t) v76);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      for (int64_t v646 = v75; v646 < v645; v646 += v76) {
        // pto: %138
        ;
        int64_t v647 = (int64_t) ((uint64_t) v474 + (uint64_t) v646);
        // pto: %139
        ;
        int64_t v648 = (int64_t) ((uint64_t) v646 + (uint64_t) v65);
        // pto: %140
        ;
        int64_t v649 = (int64_t) ((uint64_t) v474 + (uint64_t) v648);
        // pto: %flat_offset_mul
        ;
        int64_t v650 = (int64_t) ((uint64_t) v646 * (uint64_t) v69);
        // pto: %t__tmp_v224
        ;
        float v651 = v590.GetValue(v650);
        // pto: %141
        ;
        float v652 = v651 + v81;
        // pto: %143, %t__tmp_v225
        ;
        float v653 = v590.GetValue((int64_t) ((uint64_t) v650 + (uint64_t) v65));
        // pto: %144
        ;
        float v654 = v653 + v81;
        // pto: %146, %t__tmp_v226
        ;
        float v655 = v590.GetValue((int64_t) ((uint64_t) v650 + (uint64_t) v76));
        // pto: %147
        ;
        float v656 = v655 + v81;
        // pto: %149, %t__tmp_v227
        ;
        float v657 = v590.GetValue((int64_t) ((uint64_t) v650 + (uint64_t) v68));
        // pto: %150
        ;
        float v658 = v657 + v81;
        // pto: %t__tmp_v228
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v659 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %t__tmp_v228
        ;
        uint64_t v660 = (uint64_t) v56;
        TASSIGN(v659, v660);
        // pto: %151
        ;
        int64_t v661 = v647 < v75 ? v75 : v647;
        // pto: %152
        ;
        __gm__ bfloat16_t* v662 = PTOAS__GLOBAL_TENSOR_DATA(v111);
        // pto: %152
        ;
        const int64_t v663 = 0;
        // pto: %152
        ;
        const int64_t v664 = 16384;
        // pto: %152
        ;
        pto::Shape<1, 1, 1, 1, 4096> v665 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %152
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v666 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %152
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v667 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v662 + (v663 + v661 * v664), v665, v666);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
        TLOAD(v659, v667);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        // pto: %t__tmp_v229
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v668 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %t__tmp_v229
        ;
        uint64_t v669 = (uint64_t) v57;
        TASSIGN(v668, v669);
        // pto: %154
        ;
        __gm__ bfloat16_t* v670 = PTOAS__GLOBAL_TENSOR_DATA(v111);
        // pto: %154
        ;
        const int64_t v671 = 4096;
        // pto: %154
        ;
        const int64_t v672 = 16384;
        // pto: %154
        ;
        pto::Shape<1, 1, 1, 1, 4096> v673 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %154
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v674 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %154
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v675 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v670 + (v671 + v661 * v672), v673, v674);
        TLOAD(v668, v675);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %t__tmp_v230
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v676 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %t__tmp_v230
        ;
        uint64_t v677 = (uint64_t) v58;
        TASSIGN(v676, v677);
        // pto: %156
        ;
        __gm__ bfloat16_t* v678 = PTOAS__GLOBAL_TENSOR_DATA(v111);
        // pto: %156
        ;
        const int64_t v679 = 8192;
        // pto: %156
        ;
        const int64_t v680 = 16384;
        // pto: %156
        ;
        pto::Shape<1, 1, 1, 1, 4096> v681 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %156
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v682 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %156
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v683 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v678 + (v679 + v661 * v680), v681, v682);
        TLOAD(v676, v683);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %t__tmp_v231
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v684 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %t__tmp_v231
        ;
        uint64_t v685 = (uint64_t) v59;
        TASSIGN(v684, v685);
        // pto: %158
        ;
        __gm__ bfloat16_t* v686 = PTOAS__GLOBAL_TENSOR_DATA(v111);
        // pto: %158
        ;
        const int64_t v687 = 12288;
        // pto: %158
        ;
        const int64_t v688 = 16384;
        // pto: %158
        ;
        pto::Shape<1, 1, 1, 1, 4096> v689 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %158
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v690 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %158
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v691 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v686 + (v687 + v661 * v688), v689, v690);
        TLOAD(v684, v691);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %161
        ;
        int64_t v692 = (int64_t) ((uint64_t) v648 * (uint64_t) v69);
        // pto: %159
        ;
        float v693 = v590.GetValue(v692);
        // pto: %163
        ;
        float v694 = v693 + v81;
        // pto: %167, %164
        ;
        float v695 = v590.GetValue((int64_t) ((uint64_t) v692 + (uint64_t) v65));
        // pto: %168
        ;
        float v696 = v695 + v81;
        // pto: %172, %169
        ;
        float v697 = v590.GetValue((int64_t) ((uint64_t) v692 + (uint64_t) v76));
        // pto: %173
        ;
        float v698 = v697 + v81;
        // pto: %177, %174
        ;
        float v699 = v590.GetValue((int64_t) ((uint64_t) v692 + (uint64_t) v68));
        // pto: %178
        ;
        float v700 = v699 + v81;
        // pto: %4
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v701 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %4
        ;
        uint64_t v702 = (uint64_t) v35;
        TASSIGN(v701, v702);
        // pto: %179
        ;
        int64_t v703 = v649 < v75 ? v75 : v649;
        // pto: %180
        ;
        __gm__ bfloat16_t* v704 = PTOAS__GLOBAL_TENSOR_DATA(v111);
        // pto: %180
        ;
        const int64_t v705 = 0;
        // pto: %180
        ;
        const int64_t v706 = 16384;
        // pto: %180
        ;
        pto::Shape<1, 1, 1, 1, 4096> v707 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %180
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v708 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %180
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v709 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v704 + (v705 + v703 * v706), v707, v708);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v701, v709);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %5
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v710 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %5
        ;
        uint64_t v711 = (uint64_t) v60;
        TASSIGN(v710, v711);
        // pto: %182
        ;
        __gm__ bfloat16_t* v712 = PTOAS__GLOBAL_TENSOR_DATA(v111);
        // pto: %182
        ;
        const int64_t v713 = 4096;
        // pto: %182
        ;
        const int64_t v714 = 16384;
        // pto: %182
        ;
        pto::Shape<1, 1, 1, 1, 4096> v715 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %182
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v716 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %182
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v717 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v712 + (v713 + v703 * v714), v715, v716);
        TLOAD(v710, v717);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        // pto: %6
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v718 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %6
        ;
        uint64_t v719 = (uint64_t) v61;
        TASSIGN(v718, v719);
        // pto: %184
        ;
        __gm__ bfloat16_t* v720 = PTOAS__GLOBAL_TENSOR_DATA(v111);
        // pto: %184
        ;
        const int64_t v721 = 8192;
        // pto: %184
        ;
        const int64_t v722 = 16384;
        // pto: %184
        ;
        pto::Shape<1, 1, 1, 1, 4096> v723 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %184
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v724 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %184
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v725 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v720 + (v721 + v703 * v722), v723, v724);
        TLOAD(v718, v725);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        // pto: %7
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v726 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %7
        ;
        uint64_t v727 = (uint64_t) v62;
        TASSIGN(v726, v727);
        // pto: %186
        ;
        __gm__ bfloat16_t* v728 = PTOAS__GLOBAL_TENSOR_DATA(v111);
        // pto: %186
        ;
        const int64_t v729 = 12288;
        // pto: %186
        ;
        const int64_t v730 = 16384;
        // pto: %186
        ;
        pto::Shape<1, 1, 1, 1, 4096> v731 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %186
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v732 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %186
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v733 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v728 + (v729 + v703 * v730), v731, v732);
        TLOAD(v726, v733);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        // pto: %scalar_x0_inline666__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v734 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %scalar_x0_inline666__ssa_v0
        ;
        uint64_t v735 = (uint64_t) v36;
        TASSIGN(v734, v735);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        RoundMode v736 = RoundMode::CAST_ROUND;
        SaturationMode v737 = SaturationMode::ON;
        TCVT(v734, v659, v736, v737);
        // pto: %scalar_y0_inline665__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v738 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %scalar_y0_inline665__ssa_v0
        ;
        uint64_t v739 = (uint64_t) v36;
        TASSIGN(v738, v739);
        pipe_barrier(PIPE_V);
        TMULS(v738, v734, v652);
        // pto: %scalar_x1_inline664__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v740 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %scalar_x1_inline664__ssa_v0
        ;
        uint64_t v741 = (uint64_t) v56;
        TASSIGN(v740, v741);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v742 = RoundMode::CAST_ROUND;
        SaturationMode v743 = SaturationMode::ON;
        TCVT(v740, v668, v742, v743);
        // pto: %scalar_y1_inline663__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v744 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %scalar_y1_inline663__ssa_v0
        ;
        uint64_t v745 = (uint64_t) v56;
        TASSIGN(v744, v745);
        pipe_barrier(PIPE_V);
        TMULS(v744, v740, v654);
        // pto: %scalar_sum01_inline808__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v746 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %scalar_sum01_inline808__ssa_v0
        ;
        uint64_t v747 = (uint64_t) v36;
        TASSIGN(v746, v747);
        pipe_barrier(PIPE_V);
        TADD(v746, v738, v744);
        // pto: %scalar_x2_inline662__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v748 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %scalar_x2_inline662__ssa_v0
        ;
        uint64_t v749 = (uint64_t) v56;
        TASSIGN(v748, v749);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        RoundMode v750 = RoundMode::CAST_ROUND;
        SaturationMode v751 = SaturationMode::ON;
        TCVT(v748, v676, v750, v751);
        // pto: %scalar_y2_inline661__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v752 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %scalar_y2_inline661__ssa_v0
        ;
        uint64_t v753 = (uint64_t) v56;
        TASSIGN(v752, v753);
        pipe_barrier(PIPE_V);
        TMULS(v752, v748, v656);
        // pto: %scalar_x3_inline659__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v754 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %scalar_x3_inline659__ssa_v0
        ;
        uint64_t v755 = (uint64_t) v57;
        TASSIGN(v754, v755);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        RoundMode v756 = RoundMode::CAST_ROUND;
        SaturationMode v757 = SaturationMode::ON;
        TCVT(v754, v684, v756, v757);
        // pto: %scalar_y3_inline658__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v758 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %scalar_y3_inline658__ssa_v0
        ;
        uint64_t v759 = (uint64_t) v57;
        TASSIGN(v758, v759);
        pipe_barrier(PIPE_V);
        TMULS(v758, v754, v658);
        // pto: %t__tmp_v232
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v760 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %t__tmp_v232
        ;
        uint64_t v761 = (uint64_t) v56;
        TASSIGN(v760, v761);
        pipe_barrier(PIPE_V);
        TADD(v760, v752, v758);
        // pto: %scalar_result_inline805__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v762 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %scalar_result_inline805__ssa_v0
        ;
        uint64_t v763 = (uint64_t) v36;
        TASSIGN(v762, v763);
        pipe_barrier(PIPE_V);
        TADD(v762, v746, v760);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
        // pto: %scalar_bf16_inline657__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v764 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %scalar_bf16_inline657__ssa_v0
        ;
        uint64_t v765 = (uint64_t) v36;
        TASSIGN(v764, v765);
        pipe_barrier(PIPE_V);
        RoundMode v766 = RoundMode::CAST_RINT;
        SaturationMode v767 = SaturationMode::ON;
        TCVT(v764, v762, v766, v767);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        // pto: %mixed_view_inline726__ssa_v0_pview
        ;
        __gm__ bfloat16_t* v768 = PTOAS__GLOBAL_TENSOR_DATA(v172);
        // pto: %mixed_view_inline726__ssa_v0_pview
        ;
        const int64_t v769 = 0;
        // pto: %mixed_view_inline726__ssa_v0_pview
        ;
        const int64_t v770 = 4096;
        // pto: %mixed_view_inline726__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, 4096> v771 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %mixed_view_inline726__ssa_v0_pview
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v772 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %mixed_view_inline726__ssa_v0_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v773 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v768 + (v769 + v661 * v770), v771, v772);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v773, v764);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        // pto: %8
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v774 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %8
        ;
        uint64_t v775 = (uint64_t) v63;
        TASSIGN(v774, v775);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        RoundMode v776 = RoundMode::CAST_ROUND;
        SaturationMode v777 = SaturationMode::ON;
        TCVT(v774, v701, v776, v777);
        // pto: %9
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v778 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %9
        ;
        uint64_t v779 = (uint64_t) v63;
        TASSIGN(v778, v779);
        pipe_barrier(PIPE_V);
        TMULS(v778, v774, v694);
        // pto: %10
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v780 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %10
        ;
        uint64_t v781 = (uint64_t) v35;
        TASSIGN(v780, v781);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        RoundMode v782 = RoundMode::CAST_ROUND;
        SaturationMode v783 = SaturationMode::ON;
        TCVT(v780, v710, v782, v783);
        // pto: %11
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v784 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %11
        ;
        uint64_t v785 = (uint64_t) v35;
        TASSIGN(v784, v785);
        pipe_barrier(PIPE_V);
        TMULS(v784, v780, v696);
        // pto: %12
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v786 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %12
        ;
        uint64_t v787 = (uint64_t) v63;
        TASSIGN(v786, v787);
        pipe_barrier(PIPE_V);
        TADD(v786, v778, v784);
        // pto: %13
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v788 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %13
        ;
        uint64_t v789 = (uint64_t) v35;
        TASSIGN(v788, v789);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        RoundMode v790 = RoundMode::CAST_ROUND;
        SaturationMode v791 = SaturationMode::ON;
        TCVT(v788, v718, v790, v791);
        // pto: %14
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v792 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %14
        ;
        uint64_t v793 = (uint64_t) v35;
        TASSIGN(v792, v793);
        pipe_barrier(PIPE_V);
        TMULS(v792, v788, v698);
        // pto: %15
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v794 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %15
        ;
        uint64_t v795 = (uint64_t) v60;
        TASSIGN(v794, v795);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        RoundMode v796 = RoundMode::CAST_ROUND;
        SaturationMode v797 = SaturationMode::ON;
        TCVT(v794, v726, v796, v797);
        // pto: %16
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v798 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %16
        ;
        uint64_t v799 = (uint64_t) v60;
        TASSIGN(v798, v799);
        pipe_barrier(PIPE_V);
        TMULS(v798, v794, v700);
        // pto: %17
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v800 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %17
        ;
        uint64_t v801 = (uint64_t) v35;
        TASSIGN(v800, v801);
        pipe_barrier(PIPE_V);
        TADD(v800, v792, v798);
        // pto: %18
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v802 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %18
        ;
        uint64_t v803 = (uint64_t) v63;
        TASSIGN(v802, v803);
        pipe_barrier(PIPE_V);
        TADD(v802, v786, v800);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %19
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v804 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %19
        ;
        uint64_t v805 = (uint64_t) v63;
        TASSIGN(v804, v805);
        pipe_barrier(PIPE_V);
        RoundMode v806 = RoundMode::CAST_RINT;
        SaturationMode v807 = SaturationMode::ON;
        TCVT(v804, v802, v806, v807);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        // pto: %190
        ;
        __gm__ bfloat16_t* v808 = PTOAS__GLOBAL_TENSOR_DATA(v172);
        // pto: %190
        ;
        const int64_t v809 = 0;
        // pto: %190
        ;
        const int64_t v810 = 4096;
        // pto: %190
        ;
        pto::Shape<1, 1, 1, 1, 4096> v811 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %190
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v812 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %190
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v813 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v808 + (v809 + v703 * v810), v811, v812);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v813, v804);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
      };
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_S, PIPE_V, EVENT_ID1);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      // pto: %191, %192
      ;
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      if ((int64_t) ((uint64_t) v476 - (uint64_t) v645) == v65) {
        // pto: %193
        ;
        int64_t v814 = (int64_t) ((uint64_t) v474 + (uint64_t) v645);
        int64_t v815 = (int64_t) ((uint64_t) v644 * (uint64_t) v70);
        // pto: %194
        ;
        float v816 = v590.GetValue(v815);
        // pto: %197
        ;
        float v817 = v816 + v81;
        // pto: %200, %198
        ;
        float v818 = v590.GetValue((int64_t) ((uint64_t) v815 + (uint64_t) v65));
        // pto: %201
        ;
        float v819 = v818 + v81;
        // pto: %204, %202
        ;
        float v820 = v590.GetValue((int64_t) ((uint64_t) v815 + (uint64_t) v76));
        // pto: %205
        ;
        float v821 = v820 + v81;
        // pto: %208, %206
        ;
        float v822 = v590.GetValue((int64_t) ((uint64_t) v815 + (uint64_t) v68));
        // pto: %209
        ;
        float v823 = v822 + v81;
        // pto: %20
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v824 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %20
        ;
        uint64_t v825 = (uint64_t) v56;
        TASSIGN(v824, v825);
        // pto: %210
        ;
        int64_t v826 = v814 < v75 ? v75 : v814;
        // pto: %211
        ;
        __gm__ bfloat16_t* v827 = PTOAS__GLOBAL_TENSOR_DATA(v111);
        // pto: %211
        ;
        const int64_t v828 = 0;
        // pto: %211
        ;
        const int64_t v829 = 16384;
        // pto: %211
        ;
        pto::Shape<1, 1, 1, 1, 4096> v830 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %211
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v831 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %211
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v832 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v827 + (v828 + v826 * v829), v830, v831);
        TLOAD(v824, v832);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %21
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v833 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %21
        ;
        uint64_t v834 = (uint64_t) v36;
        TASSIGN(v833, v834);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v835 = RoundMode::CAST_ROUND;
        SaturationMode v836 = SaturationMode::ON;
        TCVT(v833, v824, v835, v836);
        // pto: %22
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v837 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %22
        ;
        uint64_t v838 = (uint64_t) v36;
        TASSIGN(v837, v838);
        pipe_barrier(PIPE_V);
        TMULS(v837, v833, v817);
        // pto: %23
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v839 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %23
        ;
        uint64_t v840 = (uint64_t) v57;
        TASSIGN(v839, v840);
        // pto: %213
        ;
        __gm__ bfloat16_t* v841 = PTOAS__GLOBAL_TENSOR_DATA(v111);
        // pto: %213
        ;
        const int64_t v842 = 4096;
        // pto: %213
        ;
        const int64_t v843 = 16384;
        // pto: %213
        ;
        pto::Shape<1, 1, 1, 1, 4096> v844 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %213
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v845 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %213
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v846 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v841 + (v842 + v826 * v843), v844, v845);
        TLOAD(v839, v846);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %24
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v847 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %24
        ;
        uint64_t v848 = (uint64_t) v56;
        TASSIGN(v847, v848);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v849 = RoundMode::CAST_ROUND;
        SaturationMode v850 = SaturationMode::ON;
        TCVT(v847, v839, v849, v850);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %25
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v851 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %25
        ;
        uint64_t v852 = (uint64_t) v56;
        TASSIGN(v851, v852);
        pipe_barrier(PIPE_V);
        TMULS(v851, v847, v819);
        // pto: %26
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v853 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %26
        ;
        uint64_t v854 = (uint64_t) v36;
        TASSIGN(v853, v854);
        pipe_barrier(PIPE_V);
        TADD(v853, v837, v851);
        // pto: %27
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v855 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %27
        ;
        uint64_t v856 = (uint64_t) v57;
        TASSIGN(v855, v856);
        // pto: %215
        ;
        __gm__ bfloat16_t* v857 = PTOAS__GLOBAL_TENSOR_DATA(v111);
        // pto: %215
        ;
        const int64_t v858 = 8192;
        // pto: %215
        ;
        const int64_t v859 = 16384;
        // pto: %215
        ;
        pto::Shape<1, 1, 1, 1, 4096> v860 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %215
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v861 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %215
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v862 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v857 + (v858 + v826 * v859), v860, v861);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v855, v862);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %28
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v863 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %28
        ;
        uint64_t v864 = (uint64_t) v56;
        TASSIGN(v863, v864);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v865 = RoundMode::CAST_ROUND;
        SaturationMode v866 = SaturationMode::ON;
        TCVT(v863, v855, v865, v866);
        // pto: %29
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v867 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %29
        ;
        uint64_t v868 = (uint64_t) v56;
        TASSIGN(v867, v868);
        pipe_barrier(PIPE_V);
        TMULS(v867, v863, v821);
        // pto: %30
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v869 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %30
        ;
        uint64_t v870 = (uint64_t) v63;
        TASSIGN(v869, v870);
        // pto: %217
        ;
        __gm__ bfloat16_t* v871 = PTOAS__GLOBAL_TENSOR_DATA(v111);
        // pto: %217
        ;
        const int64_t v872 = 12288;
        // pto: %217
        ;
        const int64_t v873 = 16384;
        // pto: %217
        ;
        pto::Shape<1, 1, 1, 1, 4096> v874 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %217
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v875 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %217
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v876 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v871 + (v872 + v826 * v873), v874, v875);
        TLOAD(v869, v876);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %31
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v877 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %31
        ;
        uint64_t v878 = (uint64_t) v57;
        TASSIGN(v877, v878);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v879 = RoundMode::CAST_ROUND;
        SaturationMode v880 = SaturationMode::ON;
        TCVT(v877, v869, v879, v880);
        // pto: %32
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v881 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %32
        ;
        uint64_t v882 = (uint64_t) v57;
        TASSIGN(v881, v882);
        pipe_barrier(PIPE_V);
        TMULS(v881, v877, v823);
        // pto: %33
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v883 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %33
        ;
        uint64_t v884 = (uint64_t) v56;
        TASSIGN(v883, v884);
        pipe_barrier(PIPE_V);
        TADD(v883, v867, v881);
        // pto: %34
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v885 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %34
        ;
        uint64_t v886 = (uint64_t) v36;
        TASSIGN(v885, v886);
        pipe_barrier(PIPE_V);
        TADD(v885, v853, v883);
        // pto: %35
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v887 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %35
        ;
        uint64_t v888 = (uint64_t) v36;
        TASSIGN(v887, v888);
        pipe_barrier(PIPE_V);
        RoundMode v889 = RoundMode::CAST_RINT;
        SaturationMode v890 = SaturationMode::ON;
        TCVT(v887, v885, v889, v890);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %220
        ;
        __gm__ bfloat16_t* v891 = PTOAS__GLOBAL_TENSOR_DATA(v172);
        // pto: %220
        ;
        const int64_t v892 = 0;
        // pto: %220
        ;
        const int64_t v893 = 4096;
        // pto: %220
        ;
        pto::Shape<1, 1, 1, 1, 4096> v894 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %220
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v895 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %220
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v896 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v891 + (v892 + v826 * v893), v894, v895);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v896, v887);
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