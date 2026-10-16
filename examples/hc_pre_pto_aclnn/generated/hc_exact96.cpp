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

AICORE void hc_exact96_aic(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ bfloat16_t* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, __gm__ float* v10, __gm__ float* v11, __gm__ bfloat16_t* v12, __gm__ float* v13, int64_t v14, int32_t v15, int64_t v16, int64_t v17, int64_t v18, int64_t v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int64_t v28, int64_t v29, int32_t v30, int32_t v31) {
  using T = float;

  #if defined(__DAV_CUBE__)
  const int32_t v32 = 0;
  // pto: %c0_i64
  const int64_t v33 = 0;
  // pto: %c262144_i64
  const int64_t v34 = 262144;
  // pto: %c32768_i64
  const int64_t v35 = 32768;
  // pto: %c32_index
  const int64_t v36 = 32;
  // pto: %c1_index
  const int64_t v37 = 1;
  // pto: %c24_index
  const int64_t v38 = 24;
  // pto: %c16384_index
  const int64_t v39 = 16384;
  // pto: %c8_index
  const int64_t v40 = 8;
  // pto: %c2048_index
  const int64_t v41 = 2048;
  // pto: %c0_index
  const int64_t v42 = 0;
  // pto: %c2_index
  const int64_t v43 = 2;
  // pto: %c1024_index
  const int64_t v44 = 1024;
  // pto: %c256_index
  const int64_t v45 = 256;
  // pto: %c512_index
  const int64_t v46 = 512;
  // pto: %c768_index
  const int64_t v47 = 768;
  // pto: %11
  int64_t v48 = (int64_t) ((uint64_t) v28 * (uint64_t) v40);
  // pto: %mixes_partials_inline1072__ssa_v0_view
  const int64_t v49 = 1;
  // pto: %mixes_partials_inline1072__ssa_v0_view
  const int64_t v50 = 1;
  // pto: %mixes_partials_inline1072__ssa_v0_view
  const int64_t v51 = 1;
  // pto: %mixes_partials_inline1072__ssa_v0_view
  int64_t v52 = v48 * v36;
  // pto: %mixes_partials_inline1072__ssa_v0_view
  int64_t v53 = v51 * v52;
  // pto: %mixes_partials_inline1072__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v54 = pto::Shape<1, 1, 1, -1, -1>(v49, v50, v51, v48, v36);
  // pto: %mixes_partials_inline1072__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v55 = pto::Stride<-1, -1, -1, -1, -1>(v50 * v53, v53, v52, v36, v37);
  // pto: %mixes_partials_inline1072__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v56 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v54, v55);
  // pto: %hc_fn__ssa_v0_view
  const int64_t v57 = 1;
  // pto: %hc_fn__ssa_v0_view
  const int64_t v58 = 1;
  // pto: %hc_fn__ssa_v0_view
  const int64_t v59 = 1;
  // pto: %hc_fn__ssa_v0_view
  int64_t v60 = v38 * v39;
  // pto: %hc_fn__ssa_v0_view
  int64_t v61 = v59 * v60;
  // pto: %hc_fn__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v62 = pto::Shape<1, 1, 1, -1, -1>(v57, v58, v59, v38, v39);
  // pto: %hc_fn__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v63 = pto::Stride<-1, -1, -1, -1, -1>(v58 * v61, v61, v60, v39, v37);
  // pto: %hc_fn__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v64 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v62, v63);
  const int32_t v65 = 0;
  auto v66 = TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>(v13, v65, v32);
  // pto: %worker_inline1042__ssa_v0
  // pto: %linear_workers_inline1088__ssa_v0_idx
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  for (int64_t v67 = (int64_t) v30; v67 < v14; v67 += (int64_t) v15) {
    // pto: %2
    ;
    int64_t v68 = v67 % v40;
    // pto: %4, %5, %0, %1
    ;
    int64_t v69 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v68 * (uint64_t) v16) + (uint64_t) ((int64_t) (uint64_t) (v67 / v40) * (uint64_t) v36));
    // pto: %stream_product_inline1141__ssa_v0
    ;
    Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v70 = Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v36, v36);
    // pto: %stream_product_inline1141__ssa_v0
    ;
    uint64_t v71 = (uint64_t) v33;
    TASSIGN(v70, v71);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int64_t v72 = v42; v72 < v43; v72 += v37) {
      // pto: %3, %7, %6
      ;
      int64_t v73 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v68 * (uint64_t) v41) + (uint64_t) ((int64_t) (uint64_t) v72 * (uint64_t) v44));
      // pto: %weight_chunk_inline1052__ssa_v0
      ;
      Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v74 = Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v38, v44);
      // pto: %weight_chunk_inline1052__ssa_v0
      ;
      uint64_t v75 = (uint64_t) v34;
      TASSIGN(v74, v75);
      // pto: %8
      ;
      int64_t v76 = v73 < v42 ? v42 : v73;
      // pto: %hc_fn__ssa_v0_pview
      ;
      __gm__ float* v77 = PTOAS__GLOBAL_TENSOR_DATA(v64);
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v78 = 0;
      // pto: %hc_fn__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 24, 1024> v79 = pto::Shape<1, 1, 1, 24, 1024>();
      // pto: %hc_fn__ssa_v0_pview
      ;
      pto::Stride<393216, 393216, 393216, 16384, 1> v80 = pto::Stride<393216, 393216, 393216, 16384, 1>();
      // pto: %hc_fn__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 24, 1024>, pto::Stride<393216, 393216, 393216, 16384, 1>, pto::Layout::ND> v81 = GlobalTensor<float, pto::Shape<1, 1, 1, 24, 1024>, pto::Stride<393216, 393216, 393216, 16384, 1>, pto::Layout::ND>(v77 + (v78 + v76), v79, v80);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v74, v81);
      // pto: %weight_transposed_inline1075__ssa_v0
      ;
      Tile<TileType::Mat, float, 1024, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v82 = Tile<TileType::Mat, float, 1024, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v44, v38);
      // pto: %weight_transposed_inline1075__ssa_v0
      ;
      uint64_t v83 = (uint64_t) v34;
      TASSIGN(v82, v83);
      // pto: %gathered_chunk_inline1096__ssa_v0
      ;
      const int32_t v84 = 32;
      // pto: %gathered_chunk_inline1096__ssa_v0
      ;
      const int32_t v85 = 1024;
      // pto: %gathered_chunk_inline1096__ssa_v0
      ;
      Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v86 = Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v84, v85);
      TPOP<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v66, v86);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %left0_inline1043__ssa_v0
      ;
      Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v87 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v36, v45);
      // pto: %left0_inline1043__ssa_v0
      ;
      uint64_t v88 = (uint64_t) v33;
      TASSIGN(v87, v88);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      uint16_t v89 = static_cast<uint16_t>(v42);
      uint16_t v90 = static_cast<uint16_t>(v42);
      TEXTRACT(v87, v86, v89, v90);
      // pto: %right0_inline1040__ssa_v0
      ;
      Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v91 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v45, v38);
      // pto: %right0_inline1040__ssa_v0
      ;
      uint64_t v92 = (uint64_t) v33;
      TASSIGN(v91, v92);
      uint16_t v93 = static_cast<uint16_t>(v42);
      uint16_t v94 = static_cast<uint16_t>(v42);
      TEXTRACT(v91, v82, v93, v94);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      // pto: %left1_inline1038__ssa_v0
      ;
      Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v95 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v36, v45);
      // pto: %left1_inline1038__ssa_v0
      ;
      uint64_t v96 = (uint64_t) v35;
      TASSIGN(v95, v96);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      uint16_t v97 = static_cast<uint16_t>(v42);
      uint16_t v98 = static_cast<uint16_t>(v45);
      TEXTRACT(v95, v86, v97, v98);
      // pto: %right1_inline1034__ssa_v0
      ;
      Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v99 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v45, v38);
      // pto: %right1_inline1034__ssa_v0
      ;
      uint64_t v100 = (uint64_t) v35;
      TASSIGN(v99, v100);
      uint16_t v101 = static_cast<uint16_t>(v45);
      uint16_t v102 = static_cast<uint16_t>(v42);
      TEXTRACT(v99, v82, v101, v102);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      // pto: %9
      ;
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v72 == v42) {
        pipe_barrier(PIPE_M);
        TMATMUL(v70, v87, v91);
      } else {
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v70, v70, v87, v91);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      // pto: %left2_inline1091__ssa_v0
      ;
      Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v103 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v36, v45);
      // pto: %left2_inline1091__ssa_v0
      ;
      uint64_t v104 = (uint64_t) v33;
      TASSIGN(v103, v104);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      pipe_barrier(PIPE_MTE1);
      uint16_t v105 = static_cast<uint16_t>(v42);
      uint16_t v106 = static_cast<uint16_t>(v46);
      TEXTRACT(v103, v86, v105, v106);
      // pto: %right2_inline1030__ssa_v0
      ;
      Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v107 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v45, v38);
      // pto: %right2_inline1030__ssa_v0
      ;
      uint64_t v108 = (uint64_t) v33;
      TASSIGN(v107, v108);
      uint16_t v109 = static_cast<uint16_t>(v46);
      uint16_t v110 = static_cast<uint16_t>(v42);
      TEXTRACT(v107, v82, v109, v110);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
      pipe_barrier(PIPE_M);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      TMATMUL_ACC(v70, v70, v95, v99);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
      // pto: %left3_inline1012__ssa_v0
      ;
      Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v111 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v36, v45);
      // pto: %left3_inline1012__ssa_v0
      ;
      uint64_t v112 = (uint64_t) v35;
      TASSIGN(v111, v112);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
      uint16_t v113 = static_cast<uint16_t>(v42);
      uint16_t v114 = static_cast<uint16_t>(v47);
      TEXTRACT(v111, v86, v113, v114);
      TFREE<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, TileSplitAxis::TILE_UP_DOWN>(v66);
      // pto: %right3_inline1101__ssa_v0
      ;
      Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v115 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v45, v38);
      // pto: %right3_inline1101__ssa_v0
      ;
      uint64_t v116 = (uint64_t) v35;
      TASSIGN(v115, v116);
      uint16_t v117 = static_cast<uint16_t>(v47);
      uint16_t v118 = static_cast<uint16_t>(v42);
      TEXTRACT(v115, v82, v117, v118);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      pipe_barrier(PIPE_M);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
      TMATMUL_ACC(v70, v70, v103, v107);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_M);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
      TMATMUL_ACC(v70, v70, v111, v115);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    // pto: %10
    ;
    int64_t v119 = v69 < v42 ? v42 : v69;
    // pto: %mixes_partials_inline1072__iter_v1_pview
    ;
    __gm__ float* v120 = PTOAS__GLOBAL_TENSOR_DATA(v56);
    // pto: %mixes_partials_inline1072__iter_v1_pview
    ;
    const int64_t v121 = 0;
    // pto: %mixes_partials_inline1072__iter_v1_pview
    ;
    const int64_t v122 = 32;
    // pto: %mixes_partials_inline1072__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 32, 32> v123 = pto::Shape<1, 1, 1, 32, 32>();
    // pto: %mixes_partials_inline1072__iter_v1_pview
    ;
    pto::Stride<1024, 1024, 1024, 32, 1> v124 = pto::Stride<1024, 1024, 1024, 32, 1>();
    // pto: %mixes_partials_inline1072__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v125 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v120 + (v121 + v119 * v122), v123, v124);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v125, v70);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
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

AICORE void hc_exact96_aiv(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ bfloat16_t* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, __gm__ float* v10, __gm__ float* v11, __gm__ bfloat16_t* v12, __gm__ float* v13, int64_t v14, int32_t v15, int64_t v16, int64_t v17, int64_t v18, int64_t v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int64_t v28, int64_t v29, int32_t v30, int32_t v31, int32_t v32) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c65536_i64
  const int64_t v33 = 65536;
  // pto: %c0_i64
  const int64_t v34 = 0;
  // pto: %c131072_i64
  const int64_t v35 = 131072;
  // pto: %c1024_i64
  const int64_t v36 = 1024;
  // pto: %c1536_i64
  const int64_t v37 = 1536;
  // pto: %c1088_i64
  const int64_t v38 = 1088;
  const int64_t v39 = 32;
  // pto: %c512_i64
  const int64_t v40 = 512;
  // pto: %c768_i64
  const int64_t v41 = 768;
  // pto: %c1280_i64
  const int64_t v42 = 1280;
  // pto: %c256_i64
  const int64_t v43 = 256;
  // pto: %c1792_i64
  const int64_t v44 = 1792;
  // pto: %c2048_i64
  const int64_t v45 = 2048;
  // pto: %c544_i64
  const int64_t v46 = 544;
  // pto: %c1312_i64
  const int64_t v47 = 1312;
  // pto: %c1344_i64
  const int64_t v48 = 1344;
  // pto: %c1824_i64
  const int64_t v49 = 1824;
  // pto: %c1856_i64
  const int64_t v50 = 1856;
  // pto: %c1888_i64
  const int64_t v51 = 1888;
  // pto: %c1056_i64
  const int64_t v52 = 1056;
  // pto: %c114688_i64
  const int64_t v53 = 114688;
  // pto: %c128_i64
  const int64_t v54 = 128;
  // pto: %c16384_i64
  const int64_t v55 = 16384;
  // pto: %c32768_i64
  const int64_t v56 = 32768;
  // pto: %c40960_i64
  const int64_t v57 = 40960;
  // pto: %c98304_i64
  const int64_t v58 = 98304;
  // pto: %c81920_i64
  const int64_t v59 = 81920;
  // pto: %c90112_i64
  const int64_t v60 = 90112;
  // pto: %c106496_i64
  const int64_t v61 = 106496;
  // pto: %c49152_i64
  const int64_t v62 = 49152;
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
  // pto: %c0_index
  const int64_t v73 = 0;
  // pto: %c2_index
  const int64_t v74 = 2;
  // pto: %c1024_index
  const int64_t v75 = 1024;
  // pto: %c4_index
  const int64_t v76 = 4;
  // pto: %c256_index
  const int64_t v77 = 256;
  // pto: %cst_45
  const float v78 = 6.10351563E-5f;
  // pto: %cst_46
  const float v79 = 9.99999997E-7f;
  // pto: %cst_47
  const float v80 = 0.0f;
  // pto: %c64_index
  const int64_t v81 = 64;
  // pto: %c192_index
  const int64_t v82 = 192;
  // pto: %c2_i32
  const int32_t v83 = 2;
  // pto: %c4_i32
  const int32_t v84 = 4;
  // pto: %c128_index
  const int64_t v85 = 128;
  // pto: %c5_i32
  const int32_t v86 = 5;
  // pto: %c32_i32
  const int32_t v87 = 32;
  // pto: %c64_i32
  const int32_t v88 = 64;
  // pto: %c128_i32
  const int32_t v89 = 128;
  // pto: %c19_index
  const int64_t v90 = 19;
  // pto: %cst_58
  const float v91 = 0.25f;
  // pto: %cst_59
  const float v92 = 1.0f;
  // pto: %cst_60
  const float v93 = 4.0f;
  // pto: %cst_61
  const float v94 = 2.0f;
  // pto: %214
  int64_t v95 = (int64_t) ((uint64_t) v28 * (uint64_t) v69);
  // pto: %rms_partials_inline1103__ssa_v0_view
  const int64_t v96 = 1;
  // pto: %rms_partials_inline1103__ssa_v0_view
  const int64_t v97 = 1;
  // pto: %rms_partials_inline1103__ssa_v0_view
  const int64_t v98 = 1;
  // pto: %rms_partials_inline1103__ssa_v0_view
  int64_t v99 = v95 * v64;
  // pto: %rms_partials_inline1103__ssa_v0_view
  int64_t v100 = v98 * v99;
  // pto: %rms_partials_inline1103__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v101 = pto::Shape<1, 1, 1, -1, -1>(v96, v97, v98, v95, v64);
  // pto: %rms_partials_inline1103__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v102 = pto::Stride<-1, -1, -1, -1, -1>(v97 * v100, v100, v99, v64, v95);
  // pto: %rms_partials_inline1103__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v103 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v101, v102);
  // pto: %x_flat_inline1114__ssa_v0_view
  const int64_t v104 = 1;
  // pto: %x_flat_inline1114__ssa_v0_view
  const int64_t v105 = 1;
  // pto: %x_flat_inline1114__ssa_v0_view
  const int64_t v106 = 1;
  // pto: %x_flat_inline1114__ssa_v0_view
  int64_t v107 = (int64_t) v29;
  // pto: %x_flat_inline1114__ssa_v0_view
  int64_t v108 = v107 * v66;
  // pto: %x_flat_inline1114__ssa_v0_view
  int64_t v109 = v106 * v108;
  // pto: %x_flat_inline1114__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v110 = pto::Shape<1, 1, 1, -1, -1>(v104, v105, v106, v107, v66);
  // pto: %x_flat_inline1114__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v111 = pto::Stride<-1, -1, -1, -1, -1>(v105 * v109, v109, v108, v66, v64);
  // pto: %x_flat_inline1114__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v112 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v110, v111);
  // pto: %215
  int64_t v113 = (int64_t) ((uint64_t) v28 * (uint64_t) v63);
  // pto: %parts_2d_inline1089__ssa_v0_view
  const int64_t v114 = 1;
  // pto: %parts_2d_inline1089__ssa_v0_view
  const int64_t v115 = 1;
  // pto: %parts_2d_inline1089__ssa_v0_view
  const int64_t v116 = 1;
  // pto: %parts_2d_inline1089__ssa_v0_view
  int64_t v117 = v68 * v113;
  // pto: %parts_2d_inline1089__ssa_v0_view
  int64_t v118 = v116 * v117;
  // pto: %parts_2d_inline1089__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v119 = pto::Shape<1, 1, 1, -1, -1>(v114, v115, v116, v68, v113);
  // pto: %parts_2d_inline1089__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v120 = pto::Stride<-1, -1, -1, -1, -1>(v115 * v118, v118, v117, v113, v64);
  // pto: %parts_2d_inline1089__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v121 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v119, v120);
  // pto: %rms_2d_inline1207__ssa_v0_view
  const int64_t v122 = 1;
  // pto: %rms_2d_inline1207__ssa_v0_view
  const int64_t v123 = 1;
  // pto: %rms_2d_inline1207__ssa_v0_view
  const int64_t v124 = 1;
  // pto: %rms_2d_inline1207__ssa_v0_view
  int64_t v125 = (int64_t) v28;
  // pto: %rms_2d_inline1207__ssa_v0_view
  int64_t v126 = v69 * v125;
  // pto: %rms_2d_inline1207__ssa_v0_view
  int64_t v127 = v124 * v126;
  // pto: %rms_2d_inline1207__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v128 = pto::Shape<1, 1, 1, -1, -1>(v122, v123, v124, v69, (int64_t) v28);
  // pto: %rms_2d_inline1207__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v129 = pto::Stride<-1, -1, -1, -1, -1>(v123 * v127, v127, v126, v125, v64);
  // pto: %rms_2d_inline1207__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v130 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v128, v129);
  // pto: %hc_base_2d_inline1162__ssa_v0_view
  const int64_t v131 = 1;
  // pto: %hc_base_2d_inline1162__ssa_v0_view
  const int64_t v132 = 1;
  // pto: %hc_base_2d_inline1162__ssa_v0_view
  const int64_t v133 = 1;
  // pto: %hc_base_2d_inline1162__ssa_v0_view
  int64_t v134 = v64 * v65;
  // pto: %hc_base_2d_inline1162__ssa_v0_view
  int64_t v135 = v133 * v134;
  // pto: %hc_base_2d_inline1162__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v136 = pto::Shape<1, 1, 1, -1, -1>(v131, v132, v133, v64, v65);
  // pto: %hc_base_2d_inline1162__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v137 = pto::Stride<-1, -1, -1, -1, -1>(v132 * v135, v135, v134, v65, v64);
  // pto: %hc_base_2d_inline1162__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v138 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v8, v136, v137);
  // pto: %comb_inline1025__ssa_v0_view
  const int64_t v139 = 1;
  // pto: %comb_inline1025__ssa_v0_view
  const int64_t v140 = 1;
  // pto: %comb_inline1025__ssa_v0_view
  const int64_t v141 = 1;
  // pto: %comb_inline1025__ssa_v0_view
  int64_t v142 = (int64_t) v29;
  // pto: %comb_inline1025__ssa_v0_view
  int64_t v143 = v142 * v69;
  // pto: %comb_inline1025__ssa_v0_view
  int64_t v144 = v141 * v143;
  // pto: %comb_inline1025__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v145 = pto::Shape<1, 1, 1, -1, -1>(v139, v140, v141, v142, v69);
  // pto: %comb_inline1025__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v146 = pto::Stride<-1, -1, -1, -1, -1>(v140 * v144, v144, v143, v69, v64);
  // pto: %comb_inline1025__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v147 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v9, v145, v146);
  // pto: %hc_scale_2d_inline1090__ssa_v0_view
  const int64_t v148 = 1;
  // pto: %hc_scale_2d_inline1090__ssa_v0_view
  const int64_t v149 = 1;
  // pto: %hc_scale_2d_inline1090__ssa_v0_view
  const int64_t v150 = 1;
  // pto: %hc_scale_2d_inline1090__ssa_v0_view
  int64_t v151 = v64 * v67;
  // pto: %hc_scale_2d_inline1090__ssa_v0_view
  int64_t v152 = v150 * v151;
  // pto: %hc_scale_2d_inline1090__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v153 = pto::Shape<1, 1, 1, -1, -1>(v148, v149, v150, v64, v67);
  // pto: %hc_scale_2d_inline1090__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v154 = pto::Stride<-1, -1, -1, -1, -1>(v149 * v152, v152, v151, v67, v64);
  // pto: %hc_scale_2d_inline1090__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v155 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v10, v153, v154);
  // pto: %216
  int64_t v156 = (int64_t) ((uint64_t) v29 * (uint64_t) v76);
  // pto: %post_flat_inline1069__ssa_v0_view
  const int64_t v157 = 1;
  // pto: %post_flat_inline1069__ssa_v0_view
  const int64_t v158 = 1;
  // pto: %post_flat_inline1069__ssa_v0_view
  const int64_t v159 = 1;
  // pto: %post_flat_inline1069__ssa_v0_view
  int64_t v160 = v64 * v156;
  // pto: %post_flat_inline1069__ssa_v0_view
  int64_t v161 = v159 * v160;
  // pto: %post_flat_inline1069__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v162 = pto::Shape<1, 1, 1, -1, -1>(v157, v158, v159, v64, v156);
  // pto: %post_flat_inline1069__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v163 = pto::Stride<-1, -1, -1, -1, -1>(v158 * v161, v161, v160, v156, v64);
  // pto: %post_flat_inline1069__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v164 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v11, v162, v163);
  // pto: %mixed_view_inline1172__ssa_v0_view
  const int64_t v165 = 1;
  // pto: %mixed_view_inline1172__ssa_v0_view
  const int64_t v166 = 1;
  // pto: %mixed_view_inline1172__ssa_v0_view
  const int64_t v167 = 1;
  // pto: %mixed_view_inline1172__ssa_v0_view
  int64_t v168 = (int64_t) v29;
  // pto: %mixed_view_inline1172__ssa_v0_view
  int64_t v169 = v168 * v70;
  // pto: %mixed_view_inline1172__ssa_v0_view
  int64_t v170 = v167 * v169;
  // pto: %mixed_view_inline1172__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v171 = pto::Shape<1, 1, 1, -1, -1>(v165, v166, v167, v168, v70);
  // pto: %mixed_view_inline1172__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v172 = pto::Stride<-1, -1, -1, -1, -1>(v166 * v170, v170, v169, v70, v64);
  // pto: %mixed_view_inline1172__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v173 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v12, v171, v172);
  const int32_t v174 = 0;
  auto v175 = TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>(v13, v174, v71);
  // pto: %worker_inline1042__ssa_v0
  int64_t v176 = (int64_t) v30;
  // pto: %linear_workers_inline1088__ssa_v0_idx
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  set_flag(PIPE_S, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  for (int64_t v177 = v176; v177 < v14; v177 += (int64_t) v15) {
    // pto: %34
    ;
    int64_t v178 = v177 % v68;
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    for (int64_t v179 = v73; v179 < v74; v179 += v64) {
      // pto: %35, %37, %36
      ;
      int64_t v180 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v178 * (uint64_t) v72) + (uint64_t) ((int64_t) (uint64_t) v179 * (uint64_t) v75));
      // pto: %32, %33, %39, %lane_inline1068__ssa_v0, %38
      ;
      int64_t v181 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) (v177 / v68) * (uint64_t) v63) + (uint64_t) ((int64_t) (uint64_t) ((int64_t) v32) * (uint64_t) v69));
      // pto: %stream_source_inline1140__ssa_v0
      ;
      Tile<TileType::Vec, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v182 = Tile<TileType::Vec, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
      // pto: %stream_source_inline1140__ssa_v0
      ;
      uint64_t v183 = (uint64_t) v33;
      TASSIGN(v182, v183);
      // pto: %40
      ;
      int64_t v184 = v181 < v73 ? v73 : v181;
      // pto: %41
      ;
      int64_t v185 = v180 < v73 ? v73 : v180;
      // pto: %x_flat_inline1114__ssa_v0_pview
      ;
      __gm__ bfloat16_t* v186 = PTOAS__GLOBAL_TENSOR_DATA(v112);
      // pto: %x_flat_inline1114__ssa_v0_pview
      ;
      const int64_t v187 = 0;
      // pto: %x_flat_inline1114__ssa_v0_pview
      ;
      const int64_t v188 = 16384;
      // pto: %x_flat_inline1114__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 16, 1024> v189 = pto::Shape<1, 1, 1, 16, 1024>();
      // pto: %x_flat_inline1114__ssa_v0_pview
      ;
      pto::Stride<262144, 262144, 262144, 16384, 1> v190 = pto::Stride<262144, 262144, 262144, 16384, 1>();
      // pto: %x_flat_inline1114__ssa_v0_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<262144, 262144, 262144, 16384, 1>, pto::Layout::ND> v191 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<262144, 262144, 262144, 16384, 1>, pto::Layout::ND>(v186 + (v187 + v184 * v188 + v185), v189, v190);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v182, v191);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %stream_widened_inline1124__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v192 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
      // pto: %stream_widened_inline1124__ssa_v0
      ;
      uint64_t v193 = (uint64_t) v34;
      TASSIGN(v192, v193);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      RoundMode v194 = RoundMode::CAST_ROUND;
      SaturationMode v195 = SaturationMode::ON;
      TCVT(v192, v182, v194, v195);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TPUSH<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v175, v192);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      // pto: %stream_reduce_tmp_inline1102__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v196 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
      // pto: %stream_reduce_tmp_inline1102__ssa_v0
      ;
      uint64_t v197 = (uint64_t) v33;
      TASSIGN(v196, v197);
      // pto: %t__tmp_v282
      ;
      Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v198 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v75);
      // pto: %t__tmp_v282
      ;
      uint64_t v199 = (uint64_t) v34;
      TASSIGN(v198, v199);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      TMUL(v198, v192, v192);
      // pto: %stream_sq_inline1154__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v200 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v64);
      // pto: %stream_sq_inline1154__ssa_v0
      ;
      uint64_t v201 = (uint64_t) v35;
      TASSIGN(v200, v201);
      pipe_barrier(PIPE_V);
      TROWSUM(v200, v198, v196);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      // pto: %42, %43, %44, %45
      ;
      int64_t v202 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v178 * (uint64_t) v74) + (uint64_t) v179) * (uint64_t) v16) + (uint64_t) v181);
      // pto: %46
      ;
      int64_t v203 = v202 < v73 ? v73 : v202;
      // pto: %rms_partials_inline1103__iter_v1_pview
      ;
      const int64_t v204 = 0;
      // pto: %rms_partials_inline1103__iter_v1_pview
      ;
      __gm__ float* v205 = PTOAS__GLOBAL_TENSOR_DATA(v103);
      // pto: %rms_partials_inline1103__iter_v1_pview
      ;
      const int64_t v206 = 1;
      // pto: %rms_partials_inline1103__iter_v1_pview
      ;
      const int64_t v207 = 1;
      // pto: %rms_partials_inline1103__iter_v1_pview
      ;
      const int64_t v208 = 1;
      // pto: %rms_partials_inline1103__iter_v1_pview
      ;
      int64_t v209 = v69 * v64;
      // pto: %rms_partials_inline1103__iter_v1_pview
      ;
      int64_t v210 = v208 * v209;
      // pto: %rms_partials_inline1103__iter_v1_pview
      ;
      pto::Shape<1, 1, 1, 16, 1> v211 = pto::Shape<1, 1, 1, 16, 1>(v206, v207, v208, v69, v64);
      // pto: %rms_partials_inline1103__iter_v1_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v212 = pto::Stride<-1, -1, -1, -1, -1>(v207 * v210, v210, v209, v64, v95);
      // pto: %rms_partials_inline1103__iter_v1_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v213 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v205 + (v204 + v203 * v64 + v73 * v95), v211, v212);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      TSTORE(v213, v200);
    };
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  }
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  pipe_barrier(PIPE_ALL);
  dsb(DSB_DDR);
  SYNCALL<SyncCoreType::Mix>();
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  // pto: %47, %48, %tail_lane_inline998__ssa_v0
  int64_t v214 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v176 * (uint64_t) v74) + (uint64_t) ((int64_t) v32));
  // pto: %49
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  if (v214 < v17) {
    // pto: %51
    ;
    int64_t v215 = v214 < v19 ? v214 : v19;
    // pto: %50, %52
    ;
    int64_t v216 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v214 * (uint64_t) v18) + (uint64_t) v215);
    // pto: %54
    ;
    int64_t v217 = (int64_t) ((uint64_t) v19 - (uint64_t) v214);
    // pto: %55
    ;
    int64_t v218 = v217 < v73 ? v73 : v217;
    // pto: %56
    ;
    int64_t v219 = v218 < v64 ? v218 : v64;
    // pto: %53, %57
    ;
    int64_t v220 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v216 + (uint64_t) v18) + (uint64_t) v219);
    for (int64_t v221 = v216; v221 < v220; v221 += v76) {
      // pto: %scale2_inline1108__tile
      ;
      float v222 = (v5)[v74];
      // pto: %58
      ;
      int64_t v223 = (int64_t) ((uint64_t) v220 - (uint64_t) v221);
      // pto: %59
      ;
      int64_t v224 = v223 < v76 ? v223 : v76;
      // pto: %60
      ;
      int64_t v225 = (int64_t) ((uint64_t) v224 * (uint64_t) v63);
      // pto: %c_parts_load_inline1076__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v226 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v225);
      // pto: %c_parts_load_inline1076__ssa_v0
      ;
      uint64_t v227 = (uint64_t) v34;
      TASSIGN(v226, v227);
      // pto: %61
      ;
      int64_t v228 = (int64_t) ((uint64_t) v221 * (uint64_t) v63);
      // pto: %62
      ;
      int64_t v229 = v228 < v73 ? v73 : v228;
      // pto: %parts_2d_inline1089__ssa_v0_pview
      ;
      const int64_t v230 = 0;
      // pto: %parts_2d_inline1089__ssa_v0_pview
      ;
      __gm__ float* v231 = PTOAS__GLOBAL_TENSOR_DATA(v121);
      // pto: %parts_2d_inline1089__ssa_v0_pview
      ;
      const int64_t v232 = 1;
      // pto: %parts_2d_inline1089__ssa_v0_pview
      ;
      const int64_t v233 = 1;
      // pto: %parts_2d_inline1089__ssa_v0_pview
      ;
      const int64_t v234 = 1;
      // pto: %parts_2d_inline1089__ssa_v0_pview
      ;
      int64_t v235 = v68 * v113;
      // pto: %parts_2d_inline1089__ssa_v0_pview
      ;
      int64_t v236 = v234 * v235;
      // pto: %parts_2d_inline1089__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, -1> v237 = pto::Shape<1, 1, 1, 8, -1>(v232, v233, v234, v68, v225);
      // pto: %parts_2d_inline1089__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v238 = pto::Stride<-1, -1, -1, -1, -1>(v233 * v236, v236, v235, v113, v64);
      // pto: %parts_2d_inline1089__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v239 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v231 + (v230 + v73 * v113 + v229 * v64), v237, v238);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      TLOAD(v226, v239);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %t__tmp_v283
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v240 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v68, v77);
      // pto: %t__tmp_v283
      ;
      uint64_t v241 = (uint64_t) v34;
      TASSIGN(v240, v241);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      TFILLPAD<pto::TFillPadMode::InPlace>(v240, v226);
      v240.SetValidShape(v68, v77);
      // pto: %t__tmp_v284
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v242 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v77);
      // pto: %t__tmp_v284
      ;
      uint64_t v243 = (uint64_t) v34;
      TASSIGN(v242, v243);
      pipe_barrier(PIPE_V);
      TCOLSUM(v242, v240);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
      // pto: %c_total_inline1105__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v244 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v63);
      // pto: %c_total_inline1105__ssa_v0
      ;
      uint64_t v245 = (uint64_t) v34;
      TASSIGN(v244, v245);
      // pto: %c_sq_load_inline1098__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v246 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v224);
      // pto: %c_sq_load_inline1098__ssa_v0
      ;
      uint64_t v247 = (uint64_t) v36;
      TASSIGN(v246, v247);
      // pto: %64
      ;
      int64_t v248 = v221 < v73 ? v73 : v221;
      // pto: %rms_2d_inline1207__ssa_v0_pview
      ;
      int64_t v249 = (int64_t) v28;
      // pto: %rms_2d_inline1207__ssa_v0_pview
      ;
      const int64_t v250 = 0;
      // pto: %rms_2d_inline1207__ssa_v0_pview
      ;
      __gm__ float* v251 = PTOAS__GLOBAL_TENSOR_DATA(v130);
      // pto: %rms_2d_inline1207__ssa_v0_pview
      ;
      const int64_t v252 = 1;
      // pto: %rms_2d_inline1207__ssa_v0_pview
      ;
      const int64_t v253 = 1;
      // pto: %rms_2d_inline1207__ssa_v0_pview
      ;
      const int64_t v254 = 1;
      // pto: %rms_2d_inline1207__ssa_v0_pview
      ;
      int64_t v255 = v69 * v249;
      // pto: %rms_2d_inline1207__ssa_v0_pview
      ;
      int64_t v256 = v254 * v255;
      // pto: %rms_2d_inline1207__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 16, -1> v257 = pto::Shape<1, 1, 1, 16, -1>(v252, v253, v254, v69, v224);
      // pto: %rms_2d_inline1207__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v258 = pto::Stride<-1, -1, -1, -1, -1>(v253 * v256, v256, v255, v249, v64);
      // pto: %rms_2d_inline1207__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v259 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v251 + (v250 + v73 * v249 + v248 * v64), v257, v258);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
      TLOAD(v246, v259);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %t__tmp_v285
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v260 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v69, v68);
      // pto: %t__tmp_v285
      ;
      uint64_t v261 = (uint64_t) v36;
      TASSIGN(v260, v261);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      TFILLPAD<pto::TFillPadMode::InPlace>(v260, v246);
      v260.SetValidShape(v69, v68);
      // pto: %c_rms_tmp_inline1009__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v262 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %c_rms_tmp_inline1009__ssa_v0
      ;
      uint64_t v263 = (uint64_t) v37;
      TASSIGN(v262, v263);
      // pto: %t__tmp_v286
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v264 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v286
      ;
      uint64_t v265 = (uint64_t) v36;
      TASSIGN(v264, v265);
      pipe_barrier(PIPE_V);
      TCOLSUM(v264, v260);
      // pto: %t__tmp_v287
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v266 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v287
      ;
      uint64_t v267 = (uint64_t) v36;
      TASSIGN(v266, v267);
      pipe_barrier(PIPE_V);
      TMULS(v266, v264, v78);
      // pto: %t__tmp_v288
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v268 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v288
      ;
      uint64_t v269 = (uint64_t) v36;
      TASSIGN(v268, v269);
      pipe_barrier(PIPE_V);
      TADDS(v268, v266, v79);
      // pto: %t__tmp_v289
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v270 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v289
      ;
      uint64_t v271 = (uint64_t) v38;
      TASSIGN(v270, v271);
      pipe_barrier(PIPE_V);
      TRSQRT(v270, v268, v262);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      // pto: %c_inv_inline1183__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v272 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v64);
      // pto: %c_inv_inline1183__ssa_v0
      ;
      uint64_t v273 = (uint64_t) v38;
      TASSIGN(v272, v273);
      // pto: %comb_inv_inline1123__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v274 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v64);
      // pto: %comb_inv_inline1123__ssa_v0
      ;
      uint64_t v275 = (uint64_t) v38;
      TASSIGN(v274, v275);
      // pto: %t__tmp_v290
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v276 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v69);
      // pto: %t__tmp_v290
      ;
      uint64_t v277 = (uint64_t) v39;
      TASSIGN(v276, v277);
      // pto: %slice_view
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 16, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v278;
      // pto: %slice_view
      ;
      uint64_t v279 = (uint64_t) v39;
      TASSIGN(v278, v279);
      // pto: %raw_inline1164__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v280 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v69);
      // pto: %raw_inline1164__ssa_v0
      ;
      uint64_t v281 = (uint64_t) v34;
      TASSIGN(v280, v281);
      TADDS(v280, v278, v80);
      // pto: %base_inline1193__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v282 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v69);
      // pto: %base_inline1193__ssa_v0
      ;
      uint64_t v283 = (uint64_t) v36;
      TASSIGN(v282, v283);
      // pto: %hc_base_2d_inline1162__ssa_v0_pview
      ;
      __gm__ float* v284 = PTOAS__GLOBAL_TENSOR_DATA(v138);
      // pto: %hc_base_2d_inline1162__ssa_v0_pview
      ;
      unsigned v285 = 8;
      // pto: %hc_base_2d_inline1162__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 16> v286 = pto::Shape<1, 1, 1, 1, 16>();
      // pto: %hc_base_2d_inline1162__ssa_v0_pview
      ;
      pto::Stride<24, 24, 24, 24, 1> v287 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %hc_base_2d_inline1162__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v288 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v284 + v285, v286, v287);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      TLOAD(v282, v288);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      // pto: %t__tmp_v291
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v289 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v69);
      // pto: %t__tmp_v291
      ;
      uint64_t v290 = (uint64_t) v40;
      TASSIGN(v289, v290);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v289, v280, v274);
      // pto: %t__tmp_v292
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v291 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v69);
      // pto: %t__tmp_v292
      ;
      uint64_t v292 = (uint64_t) v40;
      TASSIGN(v291, v292);
      pipe_barrier(PIPE_V);
      TMULS(v291, v289, v222);
      // pto: %t__tmp_v293
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v293 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v69);
      // pto: %t__tmp_v293
      ;
      uint64_t v294 = (uint64_t) v34;
      TASSIGN(v293, v294);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      TCOLEXPAND(v293, v282);
      // pto: %logits_inline1050__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v295 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v69);
      // pto: %logits_inline1050__ssa_v0
      ;
      uint64_t v296 = (uint64_t) v41;
      TASSIGN(v295, v296);
      pipe_barrier(PIPE_V);
      TADD(v295, v291, v293);
      // pto: %t__tmp_v294
      ;
      Tile<TileType::Vec, float, 4, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v297 = Tile<TileType::Vec, float, 4, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v69);
      // pto: %t__tmp_v294
      ;
      uint64_t v298 = (uint64_t) v41;
      TASSIGN(v297, v298);
      // pto: %logits_flat_inline1143__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v299 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %logits_flat_inline1143__ssa_v0
      ;
      uint64_t v300 = (uint64_t) v41;
      TASSIGN(v299, v300);
      // pto: %ids_inline1231__ci_tmp_v0
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v301 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v82);
      // pto: %ids_inline1231__ci_tmp_v0
      ;
      uint64_t v302 = (uint64_t) v34;
      TASSIGN(v301, v302);
      // pto: %ids_inline1231__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v303 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %ids_inline1231__ssa_v0
      ;
      uint64_t v304 = (uint64_t) v42;
      TASSIGN(v303, v304);
      // pto: %ci_tmp_view
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v305;
      TRESHAPE(v305, v301);
      // pto: %ci_dst_view
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v306;
      TRESHAPE(v306, v303);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v306, v71, v305);
      // pto: %tmp_inline1144__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v307 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %tmp_inline1144__ssa_v0
      ;
      uint64_t v308 = (uint64_t) v37;
      TASSIGN(v307, v308);
      // pto: %component_ids_inline1146__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v309 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %component_ids_inline1146__ssa_v0
      ;
      uint64_t v310 = (uint64_t) v34;
      TASSIGN(v309, v310);
      pipe_barrier(PIPE_V);
      TSHRS(v309, v303, v83);
      // pto: %t__tmp_v295
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v311 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v295
      ;
      uint64_t v312 = (uint64_t) v43;
      TASSIGN(v311, v312);
      pipe_barrier(PIPE_V);
      TSHLS(v311, v309, v83);
      // pto: %token_ids_inline1147__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v313 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %token_ids_inline1147__ssa_v0
      ;
      uint64_t v314 = (uint64_t) v43;
      TASSIGN(v313, v314);
      pipe_barrier(PIPE_V);
      TSUB(v313, v303, v311);
      // pto: %t__tmp_v296
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v315 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v296
      ;
      uint64_t v316 = (uint64_t) v43;
      TASSIGN(v315, v316);
      pipe_barrier(PIPE_V);
      TSHLS(v315, v313, v84);
      // pto: %perm_in_inline1149__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v317 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %perm_in_inline1149__ssa_v0
      ;
      uint64_t v318 = (uint64_t) v34;
      TASSIGN(v317, v318);
      pipe_barrier(PIPE_V);
      TADD(v317, v315, v309);
      // pto: %out_token_inline1152__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v319 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %out_token_inline1152__ssa_v0
      ;
      uint64_t v320 = (uint64_t) v43;
      TASSIGN(v319, v320);
      pipe_barrier(PIPE_V);
      TSHRS(v319, v303, v84);
      // pto: %t__tmp_v297
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v321 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v297
      ;
      uint64_t v322 = (uint64_t) v40;
      TASSIGN(v321, v322);
      pipe_barrier(PIPE_V);
      TSHLS(v321, v319, v84);
      // pto: %out_component_inline1167__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v323 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %out_component_inline1167__ssa_v0
      ;
      uint64_t v324 = (uint64_t) v40;
      TASSIGN(v323, v324);
      pipe_barrier(PIPE_V);
      TSUB(v323, v303, v321);
      // pto: %t__tmp_v298
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v325 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v298
      ;
      uint64_t v326 = (uint64_t) v40;
      TASSIGN(v325, v326);
      pipe_barrier(PIPE_V);
      TSHLS(v325, v323, v83);
      // pto: %perm_out_inline1053__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v327 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %perm_out_inline1053__ssa_v0
      ;
      uint64_t v328 = (uint64_t) v44;
      TASSIGN(v327, v328);
      pipe_barrier(PIPE_V);
      TADD(v327, v325, v319);
      // pto: %compact_logits_inline1145__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v329 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %compact_logits_inline1145__ssa_v0
      ;
      uint64_t v330 = (uint64_t) v45;
      TASSIGN(v329, v330);
      TGATHER(v329, v299, v317, v307);
      // pto: %ids16_inline1028__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v331 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v85);
      // pto: %ids16_inline1028__ssa_v0
      ;
      uint64_t v332 = (uint64_t) v42;
      TASSIGN(v331, v332);
      // pto: %xor_tmp16_inline1153__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v333 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v85);
      // pto: %xor_tmp16_inline1153__ssa_v0
      ;
      uint64_t v334 = (uint64_t) v41;
      TASSIGN(v333, v334);
      // pto: %block_numbers_inline1133__ci_tmp_v1
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v335 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v82);
      // pto: %block_numbers_inline1133__ci_tmp_v1
      ;
      uint64_t v336 = (uint64_t) v34;
      TASSIGN(v335, v336);
      // pto: %block_numbers_inline1133__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v337 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %block_numbers_inline1133__ssa_v0
      ;
      uint64_t v338 = (uint64_t) v36;
      TASSIGN(v337, v338);
      // pto: %66
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v339;
      TRESHAPE(v339, v335);
      // pto: %67
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v340;
      TRESHAPE(v340, v337);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v340, v71, v339);
      // pto: %block_ids_inline1155__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v341 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %block_ids_inline1155__ssa_v0
      ;
      uint64_t v342 = (uint64_t) v40;
      TASSIGN(v341, v342);
      pipe_barrier(PIPE_V);
      TSHLS(v341, v337, v86);
      // pto: %block_ids16_inline1005__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v343 = Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v69);
      // pto: %block_ids16_inline1005__ssa_v0
      ;
      uint64_t v344 = (uint64_t) v40;
      TASSIGN(v343, v344);
      // pto: %block_tmp16_inline1157__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v345 = Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v69);
      // pto: %block_tmp16_inline1157__ssa_v0
      ;
      uint64_t v346 = (uint64_t) v46;
      TASSIGN(v345, v346);
      // pto: %t__tmp_v299
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v347 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v299
      ;
      uint64_t v348 = (uint64_t) v34;
      TASSIGN(v347, v348);
      TEXPANDS(v347, v84);
      // pto: %ri1_mask_inline1163__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v349 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v85);
      // pto: %ri1_mask_inline1163__ssa_v0
      ;
      uint64_t v350 = (uint64_t) v34;
      TASSIGN(v349, v350);
      // pto: %t__tmp_v300
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v351 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v85);
      // pto: %t__tmp_v300
      ;
      uint64_t v352 = (uint64_t) v43;
      TASSIGN(v351, v352);
      pipe_barrier(PIPE_V);
      TXOR(v351, v331, v349, v333);
      // pto: %ri1_inline1166__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v353 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %ri1_inline1166__ssa_v0
      ;
      uint64_t v354 = (uint64_t) v43;
      TASSIGN(v353, v354);
      // pto: %t__tmp_v301
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v355 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v301
      ;
      uint64_t v356 = (uint64_t) v34;
      TASSIGN(v355, v356);
      pipe_barrier(PIPE_V);
      TEXPANDS(v355, v87);
      // pto: %ri2_block_mask_inline1168__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v357 = Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v69);
      // pto: %ri2_block_mask_inline1168__ssa_v0
      ;
      uint64_t v358 = (uint64_t) v34;
      TASSIGN(v357, v358);
      // pto: %t__tmp_v302
      ;
      Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v359 = Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v69);
      // pto: %t__tmp_v302
      ;
      uint64_t v360 = (uint64_t) v42;
      TASSIGN(v359, v360);
      pipe_barrier(PIPE_V);
      TXOR(v359, v343, v357, v345);
      // pto: %ri2_offset_inline1169__ssa_v0
      ;
      Tile<TileType::Vec, uint32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v361 = Tile<TileType::Vec, uint32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %ri2_offset_inline1169__ssa_v0
      ;
      uint64_t v362 = (uint64_t) v42;
      TASSIGN(v361, v362);
      // pto: %t__tmp_v303
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v363 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v303
      ;
      uint64_t v364 = (uint64_t) v34;
      TASSIGN(v363, v364);
      pipe_barrier(PIPE_V);
      TEXPANDS(v363, v88);
      // pto: %ci1_block_mask_inline1170__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v365 = Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v69);
      // pto: %ci1_block_mask_inline1170__ssa_v0
      ;
      uint64_t v366 = (uint64_t) v34;
      TASSIGN(v365, v366);
      // pto: %t__tmp_v304
      ;
      Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v367 = Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v69);
      // pto: %t__tmp_v304
      ;
      uint64_t v368 = (uint64_t) v47;
      TASSIGN(v367, v368);
      pipe_barrier(PIPE_V);
      TXOR(v367, v343, v365, v345);
      // pto: %ci1_offset_inline1171__ssa_v0
      ;
      Tile<TileType::Vec, uint32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v369 = Tile<TileType::Vec, uint32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %ci1_offset_inline1171__ssa_v0
      ;
      uint64_t v370 = (uint64_t) v47;
      TASSIGN(v369, v370);
      // pto: %t__tmp_v305
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v371 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v305
      ;
      uint64_t v372 = (uint64_t) v34;
      TASSIGN(v371, v372);
      pipe_barrier(PIPE_V);
      TEXPANDS(v371, v89);
      // pto: %ci2_block_mask_inline1059__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v373 = Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v69);
      // pto: %ci2_block_mask_inline1059__ssa_v0
      ;
      uint64_t v374 = (uint64_t) v34;
      TASSIGN(v373, v374);
      // pto: %t__tmp_v306
      ;
      Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v375 = Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v69);
      // pto: %t__tmp_v306
      ;
      uint64_t v376 = (uint64_t) v48;
      TASSIGN(v375, v376);
      pipe_barrier(PIPE_V);
      TXOR(v375, v343, v373, v345);
      // pto: %ci2_offset_inline1173__ssa_v0
      ;
      Tile<TileType::Vec, uint32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v377 = Tile<TileType::Vec, uint32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %ci2_offset_inline1173__ssa_v0
      ;
      uint64_t v378 = (uint64_t) v48;
      TASSIGN(v377, v378);
      // pto: %sink_swap_0_inline1174__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v379 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %sink_swap_0_inline1174__ssa_v0
      ;
      uint64_t v380 = (uint64_t) v34;
      TASSIGN(v379, v380);
      pipe_barrier(PIPE_V);
      TGATHER(v379, v329, v353, v307);
      // pto: %max2_inline1208__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v381 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %max2_inline1208__ssa_v0
      ;
      uint64_t v382 = (uint64_t) v34;
      TASSIGN(v381, v382);
      pipe_barrier(PIPE_V);
      TMAX(v381, v329, v379);
      // pto: %t__tmp_v307
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v383 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v307
      ;
      uint64_t v384 = (uint64_t) v40;
      TASSIGN(v383, v384);
      pipe_barrier(PIPE_V);
      TGATHERB(v383, v381, v361);
      // pto: %sink_swap_1_inline1175__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v385 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %sink_swap_1_inline1175__ssa_v0
      ;
      uint64_t v386 = (uint64_t) v40;
      TASSIGN(v385, v386);
      // pto: %68
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v387;
      // pto: %68
      ;
      uint64_t v388 = (uint64_t) v40;
      TASSIGN(v387, v388);
      // pto: %max4_inline1177__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v389 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %max4_inline1177__ssa_v0
      ;
      uint64_t v390 = (uint64_t) v34;
      TASSIGN(v389, v390);
      pipe_barrier(PIPE_V);
      TMAX(v389, v381, v387);
      // pto: %t__tmp_v308
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v391 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v308
      ;
      uint64_t v392 = (uint64_t) v34;
      TASSIGN(v391, v392);
      pipe_barrier(PIPE_V);
      TSUB(v391, v329, v389);
      // pto: %exps_inline1178__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v393 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %exps_inline1178__ssa_v0
      ;
      uint64_t v394 = (uint64_t) v34;
      TASSIGN(v393, v394);
      pipe_barrier(PIPE_V);
      TEXP(v393, v391);
      // pto: %sink_swap_2_inline1120__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v395 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %sink_swap_2_inline1120__ssa_v0
      ;
      uint64_t v396 = (uint64_t) v40;
      TASSIGN(v395, v396);
      pipe_barrier(PIPE_V);
      TGATHER(v395, v393, v353, v307);
      // pto: %sum2_inline1179__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v397 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %sum2_inline1179__ssa_v0
      ;
      uint64_t v398 = (uint64_t) v40;
      TASSIGN(v397, v398);
      pipe_barrier(PIPE_V);
      TADD(v397, v393, v395);
      // pto: %t__tmp_v309
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v399 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v309
      ;
      uint64_t v400 = (uint64_t) v41;
      TASSIGN(v399, v400);
      pipe_barrier(PIPE_V);
      TGATHERB(v399, v397, v361);
      // pto: %sink_swap_3_inline1180__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v401 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %sink_swap_3_inline1180__ssa_v0
      ;
      uint64_t v402 = (uint64_t) v41;
      TASSIGN(v401, v402);
      // pto: %69
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v403;
      // pto: %69
      ;
      uint64_t v404 = (uint64_t) v41;
      TASSIGN(v403, v404);
      // pto: %sum4_inline1185__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v405 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %sum4_inline1185__ssa_v0
      ;
      uint64_t v406 = (uint64_t) v40;
      TASSIGN(v405, v406);
      pipe_barrier(PIPE_V);
      TADD(v405, v397, v403);
      // pto: %t__tmp_v310
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v407 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v310
      ;
      uint64_t v408 = (uint64_t) v34;
      TASSIGN(v407, v408);
      pipe_barrier(PIPE_V);
      TDIV(v407, v393, v405);
      // pto: %probs_inline1188__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v409 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %probs_inline1188__ssa_v0
      ;
      uint64_t v410 = (uint64_t) v34;
      TASSIGN(v409, v410);
      pipe_barrier(PIPE_V);
      TADDS(v409, v407, v79);
      // pto: %t__tmp_v311
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v411 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v311
      ;
      uint64_t v412 = (uint64_t) v40;
      TASSIGN(v411, v412);
      pipe_barrier(PIPE_V);
      TGATHERB(v411, v409, v369);
      // pto: %sink_swap_4_inline1190__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v413 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %sink_swap_4_inline1190__ssa_v0
      ;
      uint64_t v414 = (uint64_t) v40;
      TASSIGN(v413, v414);
      // pto: %70
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v415;
      // pto: %70
      ;
      uint64_t v416 = (uint64_t) v40;
      TASSIGN(v415, v416);
      // pto: %col2_inline1218__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v417 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %col2_inline1218__ssa_v0
      ;
      uint64_t v418 = (uint64_t) v40;
      TASSIGN(v417, v418);
      pipe_barrier(PIPE_V);
      TADD(v417, v409, v415);
      // pto: %t__tmp_v312
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v419 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v312
      ;
      uint64_t v420 = (uint64_t) v41;
      TASSIGN(v419, v420);
      pipe_barrier(PIPE_V);
      TGATHERB(v419, v417, v377);
      // pto: %sink_swap_5_inline1121__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v421 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %sink_swap_5_inline1121__ssa_v0
      ;
      uint64_t v422 = (uint64_t) v41;
      TASSIGN(v421, v422);
      // pto: %71
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v423;
      // pto: %71
      ;
      uint64_t v424 = (uint64_t) v41;
      TASSIGN(v423, v424);
      // pto: %t__tmp_v313
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v425 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v313
      ;
      uint64_t v426 = (uint64_t) v40;
      TASSIGN(v425, v426);
      pipe_barrier(PIPE_V);
      TADD(v425, v417, v423);
      // pto: %col4_inline1086__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v427 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %col4_inline1086__ssa_v0
      ;
      uint64_t v428 = (uint64_t) v40;
      TASSIGN(v427, v428);
      pipe_barrier(PIPE_V);
      TADDS(v427, v425, v79);
      // pto: %cur_inline1191__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v429 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %cur_inline1191__ssa_v0
      ;
      uint64_t v430 = (uint64_t) v34;
      TASSIGN(v429, v430);
      pipe_barrier(PIPE_V);
      TDIV(v429, v409, v427);
      for (int64_t v431 = v73; v431 < v90; v431 += v64) {
        // pto: %sink_swap_6_inline1195__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v432 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %sink_swap_6_inline1195__ssa_v0
        ;
        uint64_t v433 = (uint64_t) v40;
        TASSIGN(v432, v433);
        pipe_barrier(PIPE_V);
        TGATHER(v432, v429, v353, v307);
        // pto: %row2_inline1204__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v434 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %row2_inline1204__ssa_v0
        ;
        uint64_t v435 = (uint64_t) v40;
        TASSIGN(v434, v435);
        pipe_barrier(PIPE_V);
        TADD(v434, v429, v432);
        // pto: %t__tmp_v314
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v436 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %t__tmp_v314
        ;
        uint64_t v437 = (uint64_t) v41;
        TASSIGN(v436, v437);
        pipe_barrier(PIPE_V);
        TGATHERB(v436, v434, v361);
        // pto: %sink_swap_7_inline1196__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v438 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %sink_swap_7_inline1196__ssa_v0
        ;
        uint64_t v439 = (uint64_t) v41;
        TASSIGN(v438, v439);
        // pto: %72
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v440;
        // pto: %72
        ;
        uint64_t v441 = (uint64_t) v41;
        TASSIGN(v440, v441);
        // pto: %t__tmp_v315
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v442 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %t__tmp_v315
        ;
        uint64_t v443 = (uint64_t) v40;
        TASSIGN(v442, v443);
        pipe_barrier(PIPE_V);
        TADD(v442, v434, v440);
        // pto: %row4_inline1111__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v444 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %row4_inline1111__ssa_v0
        ;
        uint64_t v445 = (uint64_t) v40;
        TASSIGN(v444, v445);
        pipe_barrier(PIPE_V);
        TADDS(v444, v442, v79);
        // pto: %norm_inline1199__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v446 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %norm_inline1199__ssa_v0
        ;
        uint64_t v447 = (uint64_t) v40;
        TASSIGN(v446, v447);
        pipe_barrier(PIPE_V);
        TDIV(v446, v429, v444);
        // pto: %t__tmp_v316
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v448 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %t__tmp_v316
        ;
        uint64_t v449 = (uint64_t) v41;
        TASSIGN(v448, v449);
        pipe_barrier(PIPE_V);
        TGATHERB(v448, v446, v369);
        // pto: %sink_swap_8_inline1002__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v450 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %sink_swap_8_inline1002__ssa_v0
        ;
        uint64_t v451 = (uint64_t) v41;
        TASSIGN(v450, v451);
        // pto: %73
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v452;
        // pto: %73
        ;
        uint64_t v453 = (uint64_t) v41;
        TASSIGN(v452, v453);
        // pto: %col2_inline1218__ssa_v3
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v454 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %col2_inline1218__ssa_v3
        ;
        uint64_t v455 = (uint64_t) v41;
        TASSIGN(v454, v455);
        pipe_barrier(PIPE_V);
        TADD(v454, v446, v452);
        // pto: %t__tmp_v317
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v456 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %t__tmp_v317
        ;
        uint64_t v457 = (uint64_t) v36;
        TASSIGN(v456, v457);
        pipe_barrier(PIPE_V);
        TGATHERB(v456, v454, v377);
        // pto: %sink_swap_9_inline1165__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v458 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %sink_swap_9_inline1165__ssa_v0
        ;
        uint64_t v459 = (uint64_t) v36;
        TASSIGN(v458, v459);
        // pto: %74
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v460;
        // pto: %74
        ;
        uint64_t v461 = (uint64_t) v36;
        TASSIGN(v460, v461);
        // pto: %t__tmp_v318
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v462 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %t__tmp_v318
        ;
        uint64_t v463 = (uint64_t) v41;
        TASSIGN(v462, v463);
        pipe_barrier(PIPE_V);
        TADD(v462, v454, v460);
        // pto: %col4_inline1086__ssa_v3
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v464 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %col4_inline1086__ssa_v3
        ;
        uint64_t v465 = (uint64_t) v41;
        TASSIGN(v464, v465);
        pipe_barrier(PIPE_V);
        TADDS(v464, v462, v79);
        // pto: %cur_inline1191__ssa_v3
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v466 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
        // pto: %cur_inline1191__ssa_v3
        ;
        uint64_t v467 = (uint64_t) v34;
        TASSIGN(v466, v467);
        pipe_barrier(PIPE_V);
        TDIV(v466, v446, v464);
      };
      // pto: %row_major_inline1200__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v468 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %row_major_inline1200__ssa_v0
      ;
      uint64_t v469 = (uint64_t) v43;
      TASSIGN(v468, v469);
      pipe_barrier(PIPE_V);
      TGATHER(v468, v429, v327, v307);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
      // pto: %t__tmp_v319
      ;
      Tile<TileType::Vec, float, 4, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v470 = Tile<TileType::Vec, float, 4, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v69);
      // pto: %t__tmp_v319
      ;
      uint64_t v471 = (uint64_t) v43;
      TASSIGN(v470, v471);
      v470.SetValidShape(v224, v69);
      // pto: %comb_inline1025__ssa_v0_pview
      ;
      const int64_t v472 = 0;
      // pto: %comb_inline1025__ssa_v0_pview
      ;
      __gm__ float* v473 = PTOAS__GLOBAL_TENSOR_DATA(v147);
      // pto: %comb_inline1025__ssa_v0_pview
      ;
      const int64_t v474 = 1;
      // pto: %comb_inline1025__ssa_v0_pview
      ;
      const int64_t v475 = 1;
      // pto: %comb_inline1025__ssa_v0_pview
      ;
      const int64_t v476 = 1;
      // pto: %comb_inline1025__ssa_v0_pview
      ;
      int64_t v477 = v224 * v69;
      // pto: %comb_inline1025__ssa_v0_pview
      ;
      int64_t v478 = v476 * v477;
      // pto: %comb_inline1025__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, 16> v479 = pto::Shape<1, 1, 1, -1, 16>(v474, v475, v476, v224, v69);
      // pto: %comb_inline1025__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v480 = pto::Stride<-1, -1, -1, -1, -1>(v475 * v478, v478, v477, v69, v64);
      // pto: %comb_inline1025__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v481 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v473 + (v472 + v248 * v69 + v73 * v64), v479, v480);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
      TSTORE(v481, v470);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    };
  } else {
    // pto: %76
    ;
    int64_t v482 = (int64_t) ((uint64_t) v214 - (uint64_t) v17);
    // pto: %78
    ;
    int64_t v483 = v482 < v21 ? v482 : v21;
    // pto: %77, %79, %80
    ;
    int64_t v484 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v482 * (uint64_t) v20) + (uint64_t) v483) * (uint64_t) v74);
    // pto: %81
    ;
    int64_t v485 = (int64_t) ((uint64_t) v21 - (uint64_t) v482);
    // pto: %82
    ;
    int64_t v486 = v485 < v73 ? v73 : v485;
    // pto: %83
    ;
    int64_t v487 = v486 < v64 ? v486 : v64;
    // pto: %86, %84, %85
    ;
    int64_t v488 = (int64_t) ((uint64_t) v484 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v20 + (uint64_t) v487) * (uint64_t) v74));
    // pto: %87
    ;
    // pto: %88
    ;
    // pto: %d_begin_inline1240__phi_v2
    ;
    int64_t v489;
    // pto: %work_begin_inline996__phi_v3
    ;
    int64_t v490;
    // pto: %work_end_inline1202__phi_v3
    ;
    int64_t v491;
    if (v23 < v24) {
      // pto: %89
      ;
      // pto: %d_worker_inline1006__phi_v2
      ;
      int64_t v492;
      // pto: %d_workers_inline1219__phi_v2
      ;
      int64_t v493;
      // pto: %pair_id_inline1039__phi_v2
      ;
      int64_t v494;
      if (v482 < v25) {
        // pto: %90
        ;
        int64_t v495 = (int64_t) ((uint64_t) v26 + (uint64_t) v64);
        // pto: %91
        ;
        // pto: %93
        ;
        v492 = v482 % v495;
        v493 = v495;
        v494 = v482 / v495;
      } else {
        // pto: %95
        ;
        int64_t v496 = (int64_t) ((uint64_t) v482 - (uint64_t) v25);
        // pto: %97, %96
        ;
        // pto: %98
        ;
        v492 = v496 % v26;
        v493 = v26;
        v494 = (int64_t) ((uint64_t) v27 + (uint64_t) (v496 / v26));
      };
      // pto: %99
      ;
      uint64_t v497 = (uint64_t) v494;
      // pto: %99
      ;
      int64_t v498 = (int64_t) (v497 * (uint64_t) v74);
      // pto: %100
      ;
      int64_t v499 = v22 < v498 ? v22 : v498;
      // pto: %101
      ;
      int64_t v500 = (int64_t) ((uint64_t) v499 + (uint64_t) v74);
      // pto: %102
      ;
      // pto: %103
      ;
      int64_t v501 = v68 / v493;
      // pto: %104
      ;
      int64_t v502 = v68 % v493;
      // pto: %105
      ;
      uint64_t v503 = (uint64_t) v492;
      // pto: %106
      ;
      bool v504 = v492 < v502;
      // pto: %106
      ;
      int64_t v505 = v504 ? v492 : v502;
      // pto: %105, %107
      ;
      v489 = (int64_t) ((uint64_t) ((int64_t) v503 * (uint64_t) v501) + (uint64_t) v505);
      v490 = v499;
      v491 = v22 < v500 ? v22 : v500;
    } else {
      v489 = v73;
      v490 = v484;
      v491 = v22 < v488 ? v22 : v488;
    };
    // pto: %108
    ;
    bool v506 = v24 <= v482;
    // pto: %work_begin_inline996__phi_v5
    ;
    int64_t v507 = v506 ? v73 : v490;
    // pto: %work_end_inline1202__phi_v5
    ;
    int64_t v508 = v506 ? v73 : v491;
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    for (int64_t v509 = v507; v509 < v508; v509 += v76) {
      // pto: %109
      ;
      int64_t v510 = (int64_t) ((uint64_t) v508 - (uint64_t) v509);
      // pto: %110
      ;
      int64_t v511 = v510 < v76 ? v510 : v76;
      // pto: %111
      ;
      int64_t v512 = (int64_t) ((uint64_t) v511 * (uint64_t) v63);
      // pto: %m_parts_load_inline1156__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v513 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v512);
      // pto: %m_parts_load_inline1156__ssa_v0
      ;
      uint64_t v514 = (uint64_t) v34;
      TASSIGN(v513, v514);
      // pto: %112
      ;
      int64_t v515 = (int64_t) ((uint64_t) v509 * (uint64_t) v63);
      // pto: %113
      ;
      int64_t v516 = v515 < v73 ? v73 : v515;
      // pto: %115
      ;
      const int64_t v517 = 0;
      // pto: %115
      ;
      __gm__ float* v518 = PTOAS__GLOBAL_TENSOR_DATA(v121);
      // pto: %115
      ;
      const int64_t v519 = 1;
      // pto: %115
      ;
      const int64_t v520 = 1;
      // pto: %115
      ;
      const int64_t v521 = 1;
      // pto: %115
      ;
      int64_t v522 = v68 * v113;
      // pto: %115
      ;
      int64_t v523 = v521 * v522;
      // pto: %115
      ;
      pto::Shape<1, 1, 1, 8, -1> v524 = pto::Shape<1, 1, 1, 8, -1>(v519, v520, v521, v68, v512);
      // pto: %115
      ;
      pto::Stride<-1, -1, -1, -1, -1> v525 = pto::Stride<-1, -1, -1, -1, -1>(v520 * v523, v523, v522, v113, v64);
      // pto: %115
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v526 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v518 + (v517 + v73 * v113 + v516 * v64), v524, v525);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      TLOAD(v513, v526);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      // pto: %t__tmp_v320
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v527 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v68, v77);
      // pto: %t__tmp_v320
      ;
      uint64_t v528 = (uint64_t) v34;
      TASSIGN(v527, v528);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      TFILLPAD<pto::TFillPadMode::InPlace>(v527, v513);
      v527.SetValidShape(v68, v77);
      // pto: %t__tmp_v321
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v529 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v77);
      // pto: %t__tmp_v321
      ;
      uint64_t v530 = (uint64_t) v34;
      TASSIGN(v529, v530);
      pipe_barrier(PIPE_V);
      TCOLSUM(v529, v527);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      // pto: %m_total_inline1250__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v531 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v63);
      // pto: %m_total_inline1250__ssa_v0
      ;
      uint64_t v532 = (uint64_t) v34;
      TASSIGN(v531, v532);
      // pto: %m_sq_load_inline1251__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v533 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v511);
      // pto: %m_sq_load_inline1251__ssa_v0
      ;
      uint64_t v534 = (uint64_t) v36;
      TASSIGN(v533, v534);
      // pto: %116
      ;
      int64_t v535 = v509 < v73 ? v73 : v509;
      // pto: %117
      ;
      int64_t v536 = (int64_t) v28;
      // pto: %117
      ;
      const int64_t v537 = 0;
      // pto: %117
      ;
      __gm__ float* v538 = PTOAS__GLOBAL_TENSOR_DATA(v130);
      // pto: %117
      ;
      const int64_t v539 = 1;
      // pto: %117
      ;
      const int64_t v540 = 1;
      // pto: %117
      ;
      const int64_t v541 = 1;
      // pto: %117
      ;
      int64_t v542 = v69 * v536;
      // pto: %117
      ;
      int64_t v543 = v541 * v542;
      // pto: %117
      ;
      pto::Shape<1, 1, 1, 16, -1> v544 = pto::Shape<1, 1, 1, 16, -1>(v539, v540, v541, v69, v511);
      // pto: %117
      ;
      pto::Stride<-1, -1, -1, -1, -1> v545 = pto::Stride<-1, -1, -1, -1, -1>(v540 * v543, v543, v542, v536, v64);
      // pto: %117
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v546 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v538 + (v537 + v73 * v536 + v535 * v64), v544, v545);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      TLOAD(v533, v546);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      // pto: %t__tmp_v322
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v547 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v69, v68);
      // pto: %t__tmp_v322
      ;
      uint64_t v548 = (uint64_t) v36;
      TASSIGN(v547, v548);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      TFILLPAD<pto::TFillPadMode::InPlace>(v547, v533);
      v547.SetValidShape(v69, v68);
      // pto: %m_rms_tmp_inline1022__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v549 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %m_rms_tmp_inline1022__ssa_v0
      ;
      uint64_t v550 = (uint64_t) v37;
      TASSIGN(v549, v550);
      // pto: %t__tmp_v323
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v551 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v323
      ;
      uint64_t v552 = (uint64_t) v36;
      TASSIGN(v551, v552);
      pipe_barrier(PIPE_V);
      TCOLSUM(v551, v547);
      // pto: %t__tmp_v324
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v553 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v324
      ;
      uint64_t v554 = (uint64_t) v36;
      TASSIGN(v553, v554);
      pipe_barrier(PIPE_V);
      TMULS(v553, v551, v78);
      // pto: %t__tmp_v325
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v555 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v325
      ;
      uint64_t v556 = (uint64_t) v36;
      TASSIGN(v555, v556);
      pipe_barrier(PIPE_V);
      TADDS(v555, v553, v79);
      // pto: %t__tmp_v326
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v557 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v326
      ;
      uint64_t v558 = (uint64_t) v44;
      TASSIGN(v557, v558);
      pipe_barrier(PIPE_V);
      TRSQRT(v557, v555, v549);
      // pto: %m_inv_inline1232__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v559 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v64);
      // pto: %m_inv_inline1232__ssa_v0
      ;
      uint64_t v560 = (uint64_t) v44;
      TASSIGN(v559, v560);
      // pto: %gate_base_inline1254__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v561 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %gate_base_inline1254__ssa_v0
      ;
      uint64_t v562 = (uint64_t) v49;
      TASSIGN(v561, v562);
      // pto: %118
      ;
      __gm__ float* v563 = PTOAS__GLOBAL_TENSOR_DATA(v138);
      // pto: %118
      ;
      pto::Shape<1, 1, 1, 1, 8> v564 = pto::Shape<1, 1, 1, 1, 8>();
      // pto: %118
      ;
      pto::Stride<24, 24, 24, 24, 1> v565 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %118
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v566 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v563, v564, v565);
      TLOAD(v561, v566);
      // pto: %gate_scales_inline1256__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v567 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v67);
      // pto: %gate_scales_inline1256__ssa_v0
      ;
      uint64_t v568 = (uint64_t) v50;
      TASSIGN(v567, v568);
      // pto: %hc_scale_2d_inline1090__ssa_v0_pview
      ;
      __gm__ float* v569 = PTOAS__GLOBAL_TENSOR_DATA(v155);
      // pto: %hc_scale_2d_inline1090__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 3> v570 = pto::Shape<1, 1, 1, 1, 3>();
      // pto: %hc_scale_2d_inline1090__ssa_v0_pview
      ;
      pto::Stride<3, 3, 3, 3, 1> v571 = pto::Stride<3, 3, 3, 3, 1>();
      // pto: %hc_scale_2d_inline1090__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND> v572 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND>(v569, v570, v571);
      TLOAD(v567, v572);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      // pto: %t__ci_tmp_v2
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v573 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v82);
      // pto: %t__ci_tmp_v2
      ;
      uint64_t v574 = (uint64_t) v36;
      TASSIGN(v573, v574);
      // pto: %t__tmp_v327
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v575 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v327
      ;
      uint64_t v576 = (uint64_t) v51;
      TASSIGN(v575, v576);
      // pto: %119
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v577;
      TRESHAPE(v577, v573);
      // pto: %120
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v578;
      TRESHAPE(v578, v575);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v578, v71, v577);
      // pto: %gate_ids_inline1087__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v579 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %gate_ids_inline1087__ssa_v0
      ;
      uint64_t v580 = (uint64_t) v36;
      TASSIGN(v579, v580);
      pipe_barrier(PIPE_V);
      RoundMode v581 = RoundMode::CAST_ROUND;
      SaturationMode v582 = SaturationMode::ON;
      TCVT(v579, v575, v581, v582);
      // pto: %t__tmp_v328
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v583 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %t__tmp_v328
      ;
      uint64_t v584 = (uint64_t) v36;
      TASSIGN(v583, v584);
      pipe_barrier(PIPE_V);
      TMULS(v583, v579, v91);
      // pto: %gate_scale_ids_inline1253__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v585 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %gate_scale_ids_inline1253__ssa_v0
      ;
      uint64_t v586 = (uint64_t) v36;
      TASSIGN(v585, v586);
      pipe_barrier(PIPE_V);
      RoundMode v587 = RoundMode::CAST_FLOOR;
      SaturationMode v588 = SaturationMode::ON;
      TCVT(v585, v583, v587, v588);
      // pto: %gate_tmp_inline1197__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v589 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %gate_tmp_inline1197__ssa_v0
      ;
      uint64_t v590 = (uint64_t) v52;
      TASSIGN(v589, v590);
      // pto: %gate_scale_vec_inline1255__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v591 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %gate_scale_vec_inline1255__ssa_v0
      ;
      uint64_t v592 = (uint64_t) v38;
      TASSIGN(v591, v592);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      TGATHER(v591, v567, v585, v589);
      // pto: %t__tmp_v329
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v593 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v68);
      // pto: %t__tmp_v329
      ;
      uint64_t v594 = (uint64_t) v34;
      TASSIGN(v593, v594);
      // pto: %121
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v595;
      // pto: %121
      ;
      uint64_t v596 = (uint64_t) v34;
      TASSIGN(v595, v596);
      // pto: %gate_raw_inline1226__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v597 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v68);
      // pto: %gate_raw_inline1226__ssa_v0
      ;
      uint64_t v598 = (uint64_t) v34;
      TASSIGN(v597, v598);
      TADDS(v597, v595, v80);
      // pto: %gate_norm_inline1058__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v599 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v68);
      // pto: %gate_norm_inline1058__ssa_v0
      ;
      uint64_t v600 = (uint64_t) v34;
      TASSIGN(v599, v600);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v599, v597, v559);
      // pto: %gate_scaled_inline995__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v601 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v68);
      // pto: %gate_scaled_inline995__ssa_v0
      ;
      uint64_t v602 = (uint64_t) v34;
      TASSIGN(v601, v602);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v601, v599, v591);
      // pto: %t__tmp_v330
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v603 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v68);
      // pto: %t__tmp_v330
      ;
      uint64_t v604 = (uint64_t) v43;
      TASSIGN(v603, v604);
      TCOLEXPAND(v603, v561);
      // pto: %gate_logits_inline1257__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v605 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v68);
      // pto: %gate_logits_inline1257__ssa_v0
      ;
      uint64_t v606 = (uint64_t) v34;
      TASSIGN(v605, v606);
      pipe_barrier(PIPE_V);
      TADD(v605, v601, v603);
      // pto: %t__tmp_v331
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v607 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v68);
      // pto: %t__tmp_v331
      ;
      uint64_t v608 = (uint64_t) v34;
      TASSIGN(v607, v608);
      pipe_barrier(PIPE_V);
      TNEG(v607, v605);
      // pto: %t__tmp_v332
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v609 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v68);
      // pto: %t__tmp_v332
      ;
      uint64_t v610 = (uint64_t) v34;
      TASSIGN(v609, v610);
      pipe_barrier(PIPE_V);
      TEXP(v609, v607);
      // pto: %t__tmp_v333
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v611 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v68);
      // pto: %t__tmp_v333
      ;
      uint64_t v612 = (uint64_t) v34;
      TASSIGN(v611, v612);
      pipe_barrier(PIPE_V);
      TADDS(v611, v609, v92);
      // pto: %gate_sigmoid_inline1258__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v613 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v68);
      // pto: %gate_sigmoid_inline1258__ssa_v0
      ;
      uint64_t v614 = (uint64_t) v53;
      TASSIGN(v613, v614);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
      TRECIP(v613, v611);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      // pto: %t__ci_tmp_v3
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v615 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v82);
      // pto: %t__ci_tmp_v3
      ;
      uint64_t v616 = (uint64_t) v34;
      TASSIGN(v615, v616);
      // pto: %t__tmp_v334
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v617 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v334
      ;
      uint64_t v618 = (uint64_t) v41;
      TASSIGN(v617, v618);
      // pto: %122
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v619;
      TRESHAPE(v619, v615);
      // pto: %123
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v620;
      TRESHAPE(v620, v617);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v620, v71, v619);
      // pto: %post_ids_float_inline1259__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v621 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %post_ids_float_inline1259__ssa_v0
      ;
      uint64_t v622 = (uint64_t) v34;
      TASSIGN(v621, v622);
      pipe_barrier(PIPE_V);
      RoundMode v623 = RoundMode::CAST_ROUND;
      SaturationMode v624 = SaturationMode::ON;
      TCVT(v621, v617, v623, v624);
      // pto: %t__tmp_v335
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v625 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v335
      ;
      uint64_t v626 = (uint64_t) v54;
      TASSIGN(v625, v626);
      pipe_barrier(PIPE_V);
      TMULS(v625, v621, v91);
      // pto: %t__tmp_v336
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v627 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v336
      ;
      uint64_t v628 = (uint64_t) v54;
      TASSIGN(v627, v628);
      pipe_barrier(PIPE_V);
      RoundMode v629 = RoundMode::CAST_FLOOR;
      SaturationMode v630 = SaturationMode::ON;
      TCVT(v627, v625, v629, v630);
      // pto: %post_rows_inline1082__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v631 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %post_rows_inline1082__ssa_v0
      ;
      uint64_t v632 = (uint64_t) v54;
      TASSIGN(v631, v632);
      pipe_barrier(PIPE_V);
      RoundMode v633 = RoundMode::CAST_ROUND;
      SaturationMode v634 = SaturationMode::ON;
      TCVT(v631, v627, v633, v634);
      // pto: %t__tmp_v337
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v635 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v337
      ;
      uint64_t v636 = (uint64_t) v54;
      TASSIGN(v635, v636);
      pipe_barrier(PIPE_V);
      TMULS(v635, v631, v93);
      // pto: %t__tmp_v338
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v637 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v338
      ;
      uint64_t v638 = (uint64_t) v34;
      TASSIGN(v637, v638);
      pipe_barrier(PIPE_V);
      TADD(v637, v621, v635);
      // pto: %t__tmp_v339
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v639 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v339
      ;
      uint64_t v640 = (uint64_t) v34;
      TASSIGN(v639, v640);
      pipe_barrier(PIPE_V);
      TADDS(v639, v637, v93);
      // pto: %post_ids_inline1225__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v641 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %post_ids_inline1225__ssa_v0
      ;
      uint64_t v642 = (uint64_t) v34;
      TASSIGN(v641, v642);
      pipe_barrier(PIPE_V);
      RoundMode v643 = RoundMode::CAST_ROUND;
      SaturationMode v644 = SaturationMode::ON;
      TCVT(v641, v639, v643, v644);
      // pto: %post_tmp_inline1223__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v645 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %post_tmp_inline1223__ssa_v0
      ;
      uint64_t v646 = (uint64_t) v54;
      TASSIGN(v645, v646);
      // pto: %t__tmp_v340
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v647 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__tmp_v340
      ;
      uint64_t v648 = (uint64_t) v53;
      TASSIGN(v647, v648);
      // pto: %post_values_inline1106__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v649 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %post_values_inline1106__ssa_v0
      ;
      uint64_t v650 = (uint64_t) v43;
      TASSIGN(v649, v650);
      pipe_barrier(PIPE_V);
      TGATHER(v649, v647, v641, v645);
      // pto: %t__tmp_v341
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v651 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v341
      ;
      uint64_t v652 = (uint64_t) v34;
      TASSIGN(v651, v652);
      pipe_barrier(PIPE_V);
      TMULS(v651, v649, v94);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      // pto: %124
      ;
      int64_t v653 = (int64_t) ((uint64_t) v511 * (uint64_t) v76);
      v651.SetValidShape(v64, v653);
      // pto: %125
      ;
      bool v654 = v489 == v73;
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      if (v654) {
        // pto: %127
        ;
        int64_t v655 = (int64_t) ((uint64_t) v509 * (uint64_t) v76);
        // pto: %128
        ;
        int64_t v656 = v655 < v73 ? v73 : v655;
        // pto: %post_flat_inline1069__ssa_v0_pview
        ;
        const int64_t v657 = 0;
        // pto: %post_flat_inline1069__ssa_v0_pview
        ;
        __gm__ float* v658 = PTOAS__GLOBAL_TENSOR_DATA(v164);
        // pto: %post_flat_inline1069__ssa_v0_pview
        ;
        const int64_t v659 = 1;
        // pto: %post_flat_inline1069__ssa_v0_pview
        ;
        const int64_t v660 = 1;
        // pto: %post_flat_inline1069__ssa_v0_pview
        ;
        const int64_t v661 = 1;
        // pto: %post_flat_inline1069__ssa_v0_pview
        ;
        int64_t v662 = v64 * v156;
        // pto: %post_flat_inline1069__ssa_v0_pview
        ;
        int64_t v663 = v661 * v662;
        // pto: %post_flat_inline1069__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, -1> v664 = pto::Shape<1, 1, 1, 1, -1>(v659, v660, v661, v64, v653);
        // pto: %post_flat_inline1069__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v665 = pto::Stride<-1, -1, -1, -1, -1>(v660 * v663, v663, v662, v156, v64);
        // pto: %post_flat_inline1069__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v666 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v658 + (v657 + v73 * v156 + v656 * v64), v664, v665);
        TSTORE(v666, v651);
      };
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      // pto: %129
      ;
      int64_t v667 = v511 / v74;
      // pto: %130
      ;
      int64_t v668 = (int64_t) ((uint64_t) v667 * (uint64_t) v74);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      for (int64_t v669 = v73; v669 < v668; v669 += v74) {
        // pto: %131
        ;
        int64_t v670 = (int64_t) ((uint64_t) v509 + (uint64_t) v669);
        // pto: %132
        ;
        int64_t v671 = (int64_t) ((uint64_t) v669 + (uint64_t) v64);
        // pto: %133
        ;
        int64_t v672 = (int64_t) ((uint64_t) v509 + (uint64_t) v671);
        // pto: %flat_offset_mul
        ;
        int64_t v673 = (int64_t) ((uint64_t) v669 * (uint64_t) v68);
        // pto: %t__tmp_v342
        ;
        float v674 = v613.GetValue(v673);
        // pto: %134
        ;
        float v675 = v674 + v79;
        // pto: %136, %t__tmp_v343
        ;
        float v676 = v613.GetValue((int64_t) ((uint64_t) v673 + (uint64_t) v64));
        // pto: %137
        ;
        float v677 = v676 + v79;
        // pto: %139, %t__tmp_v344
        ;
        float v678 = v613.GetValue((int64_t) ((uint64_t) v673 + (uint64_t) v74));
        // pto: %140
        ;
        float v679 = v678 + v79;
        // pto: %142, %t__tmp_v345
        ;
        float v680 = v613.GetValue((int64_t) ((uint64_t) v673 + (uint64_t) v67));
        // pto: %143
        ;
        float v681 = v680 + v79;
        // pto: %t__tmp_v346
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v682 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %t__tmp_v346
        ;
        uint64_t v683 = (uint64_t) v55;
        TASSIGN(v682, v683);
        // pto: %144
        ;
        int64_t v684 = v670 < v73 ? v73 : v670;
        // pto: %145
        ;
        __gm__ bfloat16_t* v685 = PTOAS__GLOBAL_TENSOR_DATA(v112);
        // pto: %145
        ;
        const int64_t v686 = 0;
        // pto: %145
        ;
        const int64_t v687 = 16384;
        // pto: %145
        ;
        pto::Shape<1, 1, 1, 1, 4096> v688 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %145
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v689 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %145
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v690 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v685 + (v686 + v684 * v687), v688, v689);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
        TLOAD(v682, v690);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        // pto: %t__tmp_v347
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v691 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %t__tmp_v347
        ;
        uint64_t v692 = (uint64_t) v56;
        TASSIGN(v691, v692);
        // pto: %147
        ;
        __gm__ bfloat16_t* v693 = PTOAS__GLOBAL_TENSOR_DATA(v112);
        // pto: %147
        ;
        const int64_t v694 = 4096;
        // pto: %147
        ;
        const int64_t v695 = 16384;
        // pto: %147
        ;
        pto::Shape<1, 1, 1, 1, 4096> v696 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %147
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v697 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %147
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v698 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v693 + (v694 + v684 * v695), v696, v697);
        TLOAD(v691, v698);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %t__tmp_v348
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v699 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %t__tmp_v348
        ;
        uint64_t v700 = (uint64_t) v57;
        TASSIGN(v699, v700);
        // pto: %149
        ;
        __gm__ bfloat16_t* v701 = PTOAS__GLOBAL_TENSOR_DATA(v112);
        // pto: %149
        ;
        const int64_t v702 = 8192;
        // pto: %149
        ;
        const int64_t v703 = 16384;
        // pto: %149
        ;
        pto::Shape<1, 1, 1, 1, 4096> v704 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %149
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v705 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %149
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v706 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v701 + (v702 + v684 * v703), v704, v705);
        TLOAD(v699, v706);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %t__tmp_v349
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v707 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %t__tmp_v349
        ;
        uint64_t v708 = (uint64_t) v58;
        TASSIGN(v707, v708);
        // pto: %151
        ;
        __gm__ bfloat16_t* v709 = PTOAS__GLOBAL_TENSOR_DATA(v112);
        // pto: %151
        ;
        const int64_t v710 = 12288;
        // pto: %151
        ;
        const int64_t v711 = 16384;
        // pto: %151
        ;
        pto::Shape<1, 1, 1, 1, 4096> v712 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %151
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v713 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %151
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v714 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v709 + (v710 + v684 * v711), v712, v713);
        TLOAD(v707, v714);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %154
        ;
        int64_t v715 = (int64_t) ((uint64_t) v671 * (uint64_t) v68);
        // pto: %152
        ;
        float v716 = v613.GetValue(v715);
        // pto: %156
        ;
        float v717 = v716 + v79;
        // pto: %160, %157
        ;
        float v718 = v613.GetValue((int64_t) ((uint64_t) v715 + (uint64_t) v64));
        // pto: %161
        ;
        float v719 = v718 + v79;
        // pto: %165, %162
        ;
        float v720 = v613.GetValue((int64_t) ((uint64_t) v715 + (uint64_t) v74));
        // pto: %166
        ;
        float v721 = v720 + v79;
        // pto: %170, %167
        ;
        float v722 = v613.GetValue((int64_t) ((uint64_t) v715 + (uint64_t) v67));
        // pto: %171
        ;
        float v723 = v722 + v79;
        // pto: %0
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v724 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %0
        ;
        uint64_t v725 = (uint64_t) v33;
        TASSIGN(v724, v725);
        // pto: %172
        ;
        int64_t v726 = v672 < v73 ? v73 : v672;
        // pto: %173
        ;
        __gm__ bfloat16_t* v727 = PTOAS__GLOBAL_TENSOR_DATA(v112);
        // pto: %173
        ;
        const int64_t v728 = 0;
        // pto: %173
        ;
        const int64_t v729 = 16384;
        // pto: %173
        ;
        pto::Shape<1, 1, 1, 1, 4096> v730 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %173
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v731 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %173
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v732 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v727 + (v728 + v726 * v729), v730, v731);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v724, v732);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %1
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v733 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %1
        ;
        uint64_t v734 = (uint64_t) v59;
        TASSIGN(v733, v734);
        // pto: %175
        ;
        __gm__ bfloat16_t* v735 = PTOAS__GLOBAL_TENSOR_DATA(v112);
        // pto: %175
        ;
        const int64_t v736 = 4096;
        // pto: %175
        ;
        const int64_t v737 = 16384;
        // pto: %175
        ;
        pto::Shape<1, 1, 1, 1, 4096> v738 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %175
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v739 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %175
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v740 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v735 + (v736 + v726 * v737), v738, v739);
        TLOAD(v733, v740);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        // pto: %2
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v741 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %2
        ;
        uint64_t v742 = (uint64_t) v60;
        TASSIGN(v741, v742);
        // pto: %177
        ;
        __gm__ bfloat16_t* v743 = PTOAS__GLOBAL_TENSOR_DATA(v112);
        // pto: %177
        ;
        const int64_t v744 = 8192;
        // pto: %177
        ;
        const int64_t v745 = 16384;
        // pto: %177
        ;
        pto::Shape<1, 1, 1, 1, 4096> v746 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %177
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v747 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %177
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v748 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v743 + (v744 + v726 * v745), v746, v747);
        TLOAD(v741, v748);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        // pto: %3
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v749 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %3
        ;
        uint64_t v750 = (uint64_t) v61;
        TASSIGN(v749, v750);
        // pto: %179
        ;
        __gm__ bfloat16_t* v751 = PTOAS__GLOBAL_TENSOR_DATA(v112);
        // pto: %179
        ;
        const int64_t v752 = 12288;
        // pto: %179
        ;
        const int64_t v753 = 16384;
        // pto: %179
        ;
        pto::Shape<1, 1, 1, 1, 4096> v754 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %179
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v755 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %179
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v756 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v751 + (v752 + v726 * v753), v754, v755);
        TLOAD(v749, v756);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        // pto: %scalar_x0_inline967__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v757 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %scalar_x0_inline967__ssa_v0
        ;
        uint64_t v758 = (uint64_t) v34;
        TASSIGN(v757, v758);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        RoundMode v759 = RoundMode::CAST_ROUND;
        SaturationMode v760 = SaturationMode::ON;
        TCVT(v757, v682, v759, v760);
        // pto: %scalar_y0_inline966__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v761 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %scalar_y0_inline966__ssa_v0
        ;
        uint64_t v762 = (uint64_t) v34;
        TASSIGN(v761, v762);
        pipe_barrier(PIPE_V);
        TMULS(v761, v757, v675);
        // pto: %scalar_x1_inline965__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v763 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %scalar_x1_inline965__ssa_v0
        ;
        uint64_t v764 = (uint64_t) v55;
        TASSIGN(v763, v764);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v765 = RoundMode::CAST_ROUND;
        SaturationMode v766 = SaturationMode::ON;
        TCVT(v763, v691, v765, v766);
        // pto: %scalar_y1_inline1093__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v767 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %scalar_y1_inline1093__ssa_v0
        ;
        uint64_t v768 = (uint64_t) v55;
        TASSIGN(v767, v768);
        pipe_barrier(PIPE_V);
        TMULS(v767, v763, v677);
        // pto: %scalar_sum01_inline964__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v769 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %scalar_sum01_inline964__ssa_v0
        ;
        uint64_t v770 = (uint64_t) v34;
        TASSIGN(v769, v770);
        pipe_barrier(PIPE_V);
        TADD(v769, v761, v767);
        // pto: %scalar_x2_inline1252__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v771 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %scalar_x2_inline1252__ssa_v0
        ;
        uint64_t v772 = (uint64_t) v55;
        TASSIGN(v771, v772);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        RoundMode v773 = RoundMode::CAST_ROUND;
        SaturationMode v774 = SaturationMode::ON;
        TCVT(v771, v699, v773, v774);
        // pto: %scalar_y2_inline987__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v775 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %scalar_y2_inline987__ssa_v0
        ;
        uint64_t v776 = (uint64_t) v55;
        TASSIGN(v775, v776);
        pipe_barrier(PIPE_V);
        TMULS(v775, v771, v679);
        // pto: %scalar_x3_inline963__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v777 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %scalar_x3_inline963__ssa_v0
        ;
        uint64_t v778 = (uint64_t) v56;
        TASSIGN(v777, v778);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        RoundMode v779 = RoundMode::CAST_ROUND;
        SaturationMode v780 = SaturationMode::ON;
        TCVT(v777, v707, v779, v780);
        // pto: %scalar_y3_inline962__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v781 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %scalar_y3_inline962__ssa_v0
        ;
        uint64_t v782 = (uint64_t) v56;
        TASSIGN(v781, v782);
        pipe_barrier(PIPE_V);
        TMULS(v781, v777, v681);
        // pto: %t__tmp_v350
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v783 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %t__tmp_v350
        ;
        uint64_t v784 = (uint64_t) v55;
        TASSIGN(v783, v784);
        pipe_barrier(PIPE_V);
        TADD(v783, v775, v781);
        // pto: %scalar_result_inline961__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v785 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %scalar_result_inline961__ssa_v0
        ;
        uint64_t v786 = (uint64_t) v34;
        TASSIGN(v785, v786);
        pipe_barrier(PIPE_V);
        TADD(v785, v769, v783);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
        // pto: %scalar_bf16_inline960__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v787 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %scalar_bf16_inline960__ssa_v0
        ;
        uint64_t v788 = (uint64_t) v34;
        TASSIGN(v787, v788);
        pipe_barrier(PIPE_V);
        RoundMode v789 = RoundMode::CAST_RINT;
        SaturationMode v790 = SaturationMode::ON;
        TCVT(v787, v785, v789, v790);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
        // pto: %mixed_view_inline1172__ssa_v0_pview
        ;
        __gm__ bfloat16_t* v791 = PTOAS__GLOBAL_TENSOR_DATA(v173);
        // pto: %mixed_view_inline1172__ssa_v0_pview
        ;
        const int64_t v792 = 0;
        // pto: %mixed_view_inline1172__ssa_v0_pview
        ;
        const int64_t v793 = 4096;
        // pto: %mixed_view_inline1172__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, 4096> v794 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %mixed_view_inline1172__ssa_v0_pview
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v795 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %mixed_view_inline1172__ssa_v0_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v796 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v791 + (v792 + v684 * v793), v794, v795);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v796, v787);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        // pto: %4
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v797 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %4
        ;
        uint64_t v798 = (uint64_t) v62;
        TASSIGN(v797, v798);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        RoundMode v799 = RoundMode::CAST_ROUND;
        SaturationMode v800 = SaturationMode::ON;
        TCVT(v797, v724, v799, v800);
        // pto: %5
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v801 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %5
        ;
        uint64_t v802 = (uint64_t) v62;
        TASSIGN(v801, v802);
        pipe_barrier(PIPE_V);
        TMULS(v801, v797, v717);
        // pto: %6
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v803 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %6
        ;
        uint64_t v804 = (uint64_t) v33;
        TASSIGN(v803, v804);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        RoundMode v805 = RoundMode::CAST_ROUND;
        SaturationMode v806 = SaturationMode::ON;
        TCVT(v803, v733, v805, v806);
        // pto: %7
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v807 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %7
        ;
        uint64_t v808 = (uint64_t) v33;
        TASSIGN(v807, v808);
        pipe_barrier(PIPE_V);
        TMULS(v807, v803, v719);
        // pto: %8
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v809 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %8
        ;
        uint64_t v810 = (uint64_t) v62;
        TASSIGN(v809, v810);
        pipe_barrier(PIPE_V);
        TADD(v809, v801, v807);
        // pto: %9
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v811 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %9
        ;
        uint64_t v812 = (uint64_t) v33;
        TASSIGN(v811, v812);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        RoundMode v813 = RoundMode::CAST_ROUND;
        SaturationMode v814 = SaturationMode::ON;
        TCVT(v811, v741, v813, v814);
        // pto: %10
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v815 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %10
        ;
        uint64_t v816 = (uint64_t) v33;
        TASSIGN(v815, v816);
        pipe_barrier(PIPE_V);
        TMULS(v815, v811, v721);
        // pto: %11
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v817 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %11
        ;
        uint64_t v818 = (uint64_t) v59;
        TASSIGN(v817, v818);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        RoundMode v819 = RoundMode::CAST_ROUND;
        SaturationMode v820 = SaturationMode::ON;
        TCVT(v817, v749, v819, v820);
        // pto: %12
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v821 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %12
        ;
        uint64_t v822 = (uint64_t) v59;
        TASSIGN(v821, v822);
        pipe_barrier(PIPE_V);
        TMULS(v821, v817, v723);
        // pto: %13
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v823 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %13
        ;
        uint64_t v824 = (uint64_t) v33;
        TASSIGN(v823, v824);
        pipe_barrier(PIPE_V);
        TADD(v823, v815, v821);
        // pto: %14
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v825 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %14
        ;
        uint64_t v826 = (uint64_t) v62;
        TASSIGN(v825, v826);
        pipe_barrier(PIPE_V);
        TADD(v825, v809, v823);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %15
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v827 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %15
        ;
        uint64_t v828 = (uint64_t) v62;
        TASSIGN(v827, v828);
        pipe_barrier(PIPE_V);
        RoundMode v829 = RoundMode::CAST_RINT;
        SaturationMode v830 = SaturationMode::ON;
        TCVT(v827, v825, v829, v830);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
        // pto: %183
        ;
        __gm__ bfloat16_t* v831 = PTOAS__GLOBAL_TENSOR_DATA(v173);
        // pto: %183
        ;
        const int64_t v832 = 0;
        // pto: %183
        ;
        const int64_t v833 = 4096;
        // pto: %183
        ;
        pto::Shape<1, 1, 1, 1, 4096> v834 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %183
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v835 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %183
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v836 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v831 + (v832 + v726 * v833), v834, v835);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v836, v827);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      };
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_S, PIPE_V, EVENT_ID1);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      // pto: %184, %185
      ;
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      if ((int64_t) ((uint64_t) v511 - (uint64_t) v668) == v64) {
        // pto: %186
        ;
        int64_t v837 = (int64_t) ((uint64_t) v509 + (uint64_t) v668);
        int64_t v838 = (int64_t) ((uint64_t) v667 * (uint64_t) v69);
        // pto: %187
        ;
        float v839 = v613.GetValue(v838);
        // pto: %190
        ;
        float v840 = v839 + v79;
        // pto: %193, %191
        ;
        float v841 = v613.GetValue((int64_t) ((uint64_t) v838 + (uint64_t) v64));
        // pto: %194
        ;
        float v842 = v841 + v79;
        // pto: %197, %195
        ;
        float v843 = v613.GetValue((int64_t) ((uint64_t) v838 + (uint64_t) v74));
        // pto: %198
        ;
        float v844 = v843 + v79;
        // pto: %201, %199
        ;
        float v845 = v613.GetValue((int64_t) ((uint64_t) v838 + (uint64_t) v67));
        // pto: %202
        ;
        float v846 = v845 + v79;
        // pto: %16
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v847 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %16
        ;
        uint64_t v848 = (uint64_t) v55;
        TASSIGN(v847, v848);
        // pto: %203
        ;
        int64_t v849 = v837 < v73 ? v73 : v837;
        // pto: %204
        ;
        __gm__ bfloat16_t* v850 = PTOAS__GLOBAL_TENSOR_DATA(v112);
        // pto: %204
        ;
        const int64_t v851 = 0;
        // pto: %204
        ;
        const int64_t v852 = 16384;
        // pto: %204
        ;
        pto::Shape<1, 1, 1, 1, 4096> v853 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %204
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v854 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %204
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v855 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v850 + (v851 + v849 * v852), v853, v854);
        TLOAD(v847, v855);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %17
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v856 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %17
        ;
        uint64_t v857 = (uint64_t) v34;
        TASSIGN(v856, v857);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v858 = RoundMode::CAST_ROUND;
        SaturationMode v859 = SaturationMode::ON;
        TCVT(v856, v847, v858, v859);
        // pto: %18
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v860 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %18
        ;
        uint64_t v861 = (uint64_t) v34;
        TASSIGN(v860, v861);
        pipe_barrier(PIPE_V);
        TMULS(v860, v856, v840);
        // pto: %19
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v862 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %19
        ;
        uint64_t v863 = (uint64_t) v56;
        TASSIGN(v862, v863);
        // pto: %206
        ;
        __gm__ bfloat16_t* v864 = PTOAS__GLOBAL_TENSOR_DATA(v112);
        // pto: %206
        ;
        const int64_t v865 = 4096;
        // pto: %206
        ;
        const int64_t v866 = 16384;
        // pto: %206
        ;
        pto::Shape<1, 1, 1, 1, 4096> v867 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %206
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v868 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %206
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v869 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v864 + (v865 + v849 * v866), v867, v868);
        TLOAD(v862, v869);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %20
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v870 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %20
        ;
        uint64_t v871 = (uint64_t) v55;
        TASSIGN(v870, v871);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v872 = RoundMode::CAST_ROUND;
        SaturationMode v873 = SaturationMode::ON;
        TCVT(v870, v862, v872, v873);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %21
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v874 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %21
        ;
        uint64_t v875 = (uint64_t) v55;
        TASSIGN(v874, v875);
        pipe_barrier(PIPE_V);
        TMULS(v874, v870, v842);
        // pto: %22
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v876 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %22
        ;
        uint64_t v877 = (uint64_t) v34;
        TASSIGN(v876, v877);
        pipe_barrier(PIPE_V);
        TADD(v876, v860, v874);
        // pto: %23
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v878 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %23
        ;
        uint64_t v879 = (uint64_t) v56;
        TASSIGN(v878, v879);
        // pto: %208
        ;
        __gm__ bfloat16_t* v880 = PTOAS__GLOBAL_TENSOR_DATA(v112);
        // pto: %208
        ;
        const int64_t v881 = 8192;
        // pto: %208
        ;
        const int64_t v882 = 16384;
        // pto: %208
        ;
        pto::Shape<1, 1, 1, 1, 4096> v883 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %208
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v884 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %208
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v885 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v880 + (v881 + v849 * v882), v883, v884);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v878, v885);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %24
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v886 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %24
        ;
        uint64_t v887 = (uint64_t) v55;
        TASSIGN(v886, v887);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v888 = RoundMode::CAST_ROUND;
        SaturationMode v889 = SaturationMode::ON;
        TCVT(v886, v878, v888, v889);
        // pto: %25
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v890 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %25
        ;
        uint64_t v891 = (uint64_t) v55;
        TASSIGN(v890, v891);
        pipe_barrier(PIPE_V);
        TMULS(v890, v886, v844);
        // pto: %26
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v892 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %26
        ;
        uint64_t v893 = (uint64_t) v62;
        TASSIGN(v892, v893);
        // pto: %210
        ;
        __gm__ bfloat16_t* v894 = PTOAS__GLOBAL_TENSOR_DATA(v112);
        // pto: %210
        ;
        const int64_t v895 = 12288;
        // pto: %210
        ;
        const int64_t v896 = 16384;
        // pto: %210
        ;
        pto::Shape<1, 1, 1, 1, 4096> v897 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %210
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v898 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %210
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v899 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v894 + (v895 + v849 * v896), v897, v898);
        TLOAD(v892, v899);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %27
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v900 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %27
        ;
        uint64_t v901 = (uint64_t) v56;
        TASSIGN(v900, v901);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v902 = RoundMode::CAST_ROUND;
        SaturationMode v903 = SaturationMode::ON;
        TCVT(v900, v892, v902, v903);
        // pto: %28
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v904 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %28
        ;
        uint64_t v905 = (uint64_t) v56;
        TASSIGN(v904, v905);
        pipe_barrier(PIPE_V);
        TMULS(v904, v900, v846);
        // pto: %29
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v906 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %29
        ;
        uint64_t v907 = (uint64_t) v55;
        TASSIGN(v906, v907);
        pipe_barrier(PIPE_V);
        TADD(v906, v890, v904);
        // pto: %30
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v908 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %30
        ;
        uint64_t v909 = (uint64_t) v34;
        TASSIGN(v908, v909);
        pipe_barrier(PIPE_V);
        TADD(v908, v876, v906);
        // pto: %31
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v910 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v70);
        // pto: %31
        ;
        uint64_t v911 = (uint64_t) v34;
        TASSIGN(v910, v911);
        pipe_barrier(PIPE_V);
        RoundMode v912 = RoundMode::CAST_RINT;
        SaturationMode v913 = SaturationMode::ON;
        TCVT(v910, v908, v912, v913);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        // pto: %213
        ;
        __gm__ bfloat16_t* v914 = PTOAS__GLOBAL_TENSOR_DATA(v173);
        // pto: %213
        ;
        const int64_t v915 = 0;
        // pto: %213
        ;
        const int64_t v916 = 4096;
        // pto: %213
        ;
        pto::Shape<1, 1, 1, 1, 4096> v917 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %213
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v918 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %213
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v919 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v914 + (v915 + v849 * v916), v917, v918);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        TSTORE(v919, v910);
      };
      set_flag(PIPE_S, PIPE_V, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    };
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}