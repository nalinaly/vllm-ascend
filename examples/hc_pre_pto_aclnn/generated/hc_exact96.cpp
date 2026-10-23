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
  // pto: %mixes_partials_inline1111__ssa_v0_view
  const int64_t v49 = 1;
  // pto: %mixes_partials_inline1111__ssa_v0_view
  const int64_t v50 = 1;
  // pto: %mixes_partials_inline1111__ssa_v0_view
  const int64_t v51 = 1;
  // pto: %mixes_partials_inline1111__ssa_v0_view
  int64_t v52 = v48 * v36;
  // pto: %mixes_partials_inline1111__ssa_v0_view
  int64_t v53 = v51 * v52;
  // pto: %mixes_partials_inline1111__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v54 = pto::Shape<1, 1, 1, -1, -1>(v49, v50, v51, v48, v36);
  // pto: %mixes_partials_inline1111__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v55 = pto::Stride<-1, -1, -1, -1, -1>(v50 * v53, v53, v52, v36, v37);
  // pto: %mixes_partials_inline1111__ssa_v0_view
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
  // pto: %worker_inline1090__ssa_v0
  // pto: %linear_workers_inline1120__ssa_v0_idx
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
    // pto: %stream_product_inline1169__ssa_v0
    ;
    Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v70 = Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v36, v36);
    // pto: %stream_product_inline1169__ssa_v0
    ;
    uint64_t v71 = (uint64_t) v33;
    TASSIGN(v70, v71);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int64_t v72 = v42; v72 < v43; v72 += v37) {
      // pto: %3, %7, %6
      ;
      int64_t v73 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v68 * (uint64_t) v41) + (uint64_t) ((int64_t) (uint64_t) v72 * (uint64_t) v44));
      // pto: %weight_chunk_inline1140__ssa_v0
      ;
      Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v74 = Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v38, v44);
      // pto: %weight_chunk_inline1140__ssa_v0
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
      // pto: %weight_transposed_inline1191__ssa_v0
      ;
      Tile<TileType::Mat, float, 1024, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v82 = Tile<TileType::Mat, float, 1024, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v44, v38);
      // pto: %weight_transposed_inline1191__ssa_v0
      ;
      uint64_t v83 = (uint64_t) v34;
      TASSIGN(v82, v83);
      // pto: %gathered_chunk_inline1045__ssa_v0
      ;
      const int32_t v84 = 32;
      // pto: %gathered_chunk_inline1045__ssa_v0
      ;
      const int32_t v85 = 1024;
      // pto: %gathered_chunk_inline1045__ssa_v0
      ;
      Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v86 = Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v84, v85);
      TPOP<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v66, v86);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %left0_inline1054__ssa_v0
      ;
      Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v87 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v36, v45);
      // pto: %left0_inline1054__ssa_v0
      ;
      uint64_t v88 = (uint64_t) v33;
      TASSIGN(v87, v88);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      uint16_t v89 = static_cast<uint16_t>(v42);
      uint16_t v90 = static_cast<uint16_t>(v42);
      TEXTRACT(v87, v86, v89, v90);
      // pto: %right0_inline1070__ssa_v0
      ;
      Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v91 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v45, v38);
      // pto: %right0_inline1070__ssa_v0
      ;
      uint64_t v92 = (uint64_t) v33;
      TASSIGN(v91, v92);
      uint16_t v93 = static_cast<uint16_t>(v42);
      uint16_t v94 = static_cast<uint16_t>(v42);
      TEXTRACT(v91, v82, v93, v94);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      // pto: %left1_inline1066__ssa_v0
      ;
      Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v95 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v36, v45);
      // pto: %left1_inline1066__ssa_v0
      ;
      uint64_t v96 = (uint64_t) v35;
      TASSIGN(v95, v96);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      uint16_t v97 = static_cast<uint16_t>(v42);
      uint16_t v98 = static_cast<uint16_t>(v45);
      TEXTRACT(v95, v86, v97, v98);
      // pto: %right1_inline1142__ssa_v0
      ;
      Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v99 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v45, v38);
      // pto: %right1_inline1142__ssa_v0
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
      // pto: %left2_inline1124__ssa_v0
      ;
      Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v103 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v36, v45);
      // pto: %left2_inline1124__ssa_v0
      ;
      uint64_t v104 = (uint64_t) v33;
      TASSIGN(v103, v104);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      pipe_barrier(PIPE_MTE1);
      uint16_t v105 = static_cast<uint16_t>(v42);
      uint16_t v106 = static_cast<uint16_t>(v46);
      TEXTRACT(v103, v86, v105, v106);
      // pto: %right2_inline1027__ssa_v0
      ;
      Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v107 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v45, v38);
      // pto: %right2_inline1027__ssa_v0
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
      // pto: %left3_inline1038__ssa_v0
      ;
      Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v111 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v36, v45);
      // pto: %left3_inline1038__ssa_v0
      ;
      uint64_t v112 = (uint64_t) v35;
      TASSIGN(v111, v112);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
      uint16_t v113 = static_cast<uint16_t>(v42);
      uint16_t v114 = static_cast<uint16_t>(v47);
      TEXTRACT(v111, v86, v113, v114);
      TFREE<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, TileSplitAxis::TILE_UP_DOWN>(v66);
      // pto: %right3_inline1103__ssa_v0
      ;
      Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v115 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v45, v38);
      // pto: %right3_inline1103__ssa_v0
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
    // pto: %mixes_partials_inline1111__iter_v1_pview
    ;
    __gm__ float* v120 = PTOAS__GLOBAL_TENSOR_DATA(v56);
    // pto: %mixes_partials_inline1111__iter_v1_pview
    ;
    const int64_t v121 = 0;
    // pto: %mixes_partials_inline1111__iter_v1_pview
    ;
    const int64_t v122 = 32;
    // pto: %mixes_partials_inline1111__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 32, 32> v123 = pto::Shape<1, 1, 1, 32, 32>();
    // pto: %mixes_partials_inline1111__iter_v1_pview
    ;
    pto::Stride<1024, 1024, 1024, 32, 1> v124 = pto::Stride<1024, 1024, 1024, 32, 1>();
    // pto: %mixes_partials_inline1111__iter_v1_pview
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
  // pto: %c8192_i64
  const int64_t v36 = 8192;
  // pto: %c1024_i64
  const int64_t v37 = 1024;
  // pto: %c1536_i64
  const int64_t v38 = 1536;
  // pto: %c2048_i64
  const int64_t v39 = 2048;
  // pto: %c1088_i64
  const int64_t v40 = 1088;
  const int64_t v41 = 32;
  // pto: %c512_i64
  const int64_t v42 = 512;
  // pto: %c768_i64
  const int64_t v43 = 768;
  // pto: %c1280_i64
  const int64_t v44 = 1280;
  // pto: %c256_i64
  const int64_t v45 = 256;
  // pto: %c1792_i64
  const int64_t v46 = 1792;
  // pto: %c544_i64
  const int64_t v47 = 544;
  // pto: %c1312_i64
  const int64_t v48 = 1312;
  // pto: %c1344_i64
  const int64_t v49 = 1344;
  // pto: %c1824_i64
  const int64_t v50 = 1824;
  // pto: %c1856_i64
  const int64_t v51 = 1856;
  // pto: %c1888_i64
  const int64_t v52 = 1888;
  // pto: %c1056_i64
  const int64_t v53 = 1056;
  // pto: %c114688_i64
  const int64_t v54 = 114688;
  // pto: %c128_i64
  const int64_t v55 = 128;
  // pto: %c16384_i64
  const int64_t v56 = 16384;
  // pto: %c32768_i64
  const int64_t v57 = 32768;
  // pto: %c40960_i64
  const int64_t v58 = 40960;
  // pto: %c98304_i64
  const int64_t v59 = 98304;
  // pto: %c81920_i64
  const int64_t v60 = 81920;
  // pto: %c90112_i64
  const int64_t v61 = 90112;
  // pto: %c106496_i64
  const int64_t v62 = 106496;
  // pto: %c49152_i64
  const int64_t v63 = 49152;
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
  // pto: %cst_46
  const float v79 = 6.10351563E-5f;
  // pto: %cst_47
  const float v80 = 9.99999997E-7f;
  // pto: %cst_48
  const float v81 = 0.0f;
  // pto: %c64_index
  const int64_t v82 = 64;
  // pto: %c192_index
  const int64_t v83 = 192;
  // pto: %c2_i32
  const int32_t v84 = 2;
  // pto: %c4_i32
  const int32_t v85 = 4;
  // pto: %c128_index
  const int64_t v86 = 128;
  // pto: %c5_i32
  const int32_t v87 = 5;
  // pto: %c32_i32
  const int32_t v88 = 32;
  // pto: %c64_i32
  const int32_t v89 = 64;
  // pto: %c128_i32
  const int32_t v90 = 128;
  // pto: %c19_index
  const int64_t v91 = 19;
  // pto: %cst_59
  const float v92 = 0.25f;
  // pto: %cst_60
  const float v93 = 1.0f;
  // pto: %cst_61
  const float v94 = 4.0f;
  // pto: %cst_62
  const float v95 = 2.0f;
  // pto: %223
  int64_t v96 = (int64_t) ((uint64_t) v28 * (uint64_t) v70);
  // pto: %rms_partials_inline1178__ssa_v0_view
  const int64_t v97 = 1;
  // pto: %rms_partials_inline1178__ssa_v0_view
  const int64_t v98 = 1;
  // pto: %rms_partials_inline1178__ssa_v0_view
  const int64_t v99 = 1;
  // pto: %rms_partials_inline1178__ssa_v0_view
  int64_t v100 = v96 * v65;
  // pto: %rms_partials_inline1178__ssa_v0_view
  int64_t v101 = v99 * v100;
  // pto: %rms_partials_inline1178__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v102 = pto::Shape<1, 1, 1, -1, -1>(v97, v98, v99, v96, v65);
  // pto: %rms_partials_inline1178__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v103 = pto::Stride<-1, -1, -1, -1, -1>(v98 * v101, v101, v100, v65, v96);
  // pto: %rms_partials_inline1178__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v104 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v102, v103);
  // pto: %x_flat_inline1075__ssa_v0_view
  const int64_t v105 = 1;
  // pto: %x_flat_inline1075__ssa_v0_view
  const int64_t v106 = 1;
  // pto: %x_flat_inline1075__ssa_v0_view
  const int64_t v107 = 1;
  // pto: %x_flat_inline1075__ssa_v0_view
  int64_t v108 = (int64_t) v29;
  // pto: %x_flat_inline1075__ssa_v0_view
  int64_t v109 = v108 * v67;
  // pto: %x_flat_inline1075__ssa_v0_view
  int64_t v110 = v107 * v109;
  // pto: %x_flat_inline1075__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v111 = pto::Shape<1, 1, 1, -1, -1>(v105, v106, v107, v108, v67);
  // pto: %x_flat_inline1075__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v112 = pto::Stride<-1, -1, -1, -1, -1>(v106 * v110, v110, v109, v67, v65);
  // pto: %x_flat_inline1075__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v113 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v111, v112);
  // pto: %224
  int64_t v114 = (int64_t) ((uint64_t) v28 * (uint64_t) v64);
  // pto: %parts_2d_inline1150__ssa_v0_view
  const int64_t v115 = 1;
  // pto: %parts_2d_inline1150__ssa_v0_view
  const int64_t v116 = 1;
  // pto: %parts_2d_inline1150__ssa_v0_view
  const int64_t v117 = 1;
  // pto: %parts_2d_inline1150__ssa_v0_view
  int64_t v118 = v69 * v114;
  // pto: %parts_2d_inline1150__ssa_v0_view
  int64_t v119 = v117 * v118;
  // pto: %parts_2d_inline1150__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v120 = pto::Shape<1, 1, 1, -1, -1>(v115, v116, v117, v69, v114);
  // pto: %parts_2d_inline1150__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v121 = pto::Stride<-1, -1, -1, -1, -1>(v116 * v119, v119, v118, v114, v65);
  // pto: %parts_2d_inline1150__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v122 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v120, v121);
  // pto: %rms_2d_inline1060__ssa_v0_view
  const int64_t v123 = 1;
  // pto: %rms_2d_inline1060__ssa_v0_view
  const int64_t v124 = 1;
  // pto: %rms_2d_inline1060__ssa_v0_view
  const int64_t v125 = 1;
  // pto: %rms_2d_inline1060__ssa_v0_view
  int64_t v126 = (int64_t) v28;
  // pto: %rms_2d_inline1060__ssa_v0_view
  int64_t v127 = v70 * v126;
  // pto: %rms_2d_inline1060__ssa_v0_view
  int64_t v128 = v125 * v127;
  // pto: %rms_2d_inline1060__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v129 = pto::Shape<1, 1, 1, -1, -1>(v123, v124, v125, v70, (int64_t) v28);
  // pto: %rms_2d_inline1060__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v130 = pto::Stride<-1, -1, -1, -1, -1>(v124 * v128, v128, v127, v126, v65);
  // pto: %rms_2d_inline1060__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v131 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v129, v130);
  // pto: %hc_base_2d_inline1077__ssa_v0_view
  const int64_t v132 = 1;
  // pto: %hc_base_2d_inline1077__ssa_v0_view
  const int64_t v133 = 1;
  // pto: %hc_base_2d_inline1077__ssa_v0_view
  const int64_t v134 = 1;
  // pto: %hc_base_2d_inline1077__ssa_v0_view
  int64_t v135 = v65 * v66;
  // pto: %hc_base_2d_inline1077__ssa_v0_view
  int64_t v136 = v134 * v135;
  // pto: %hc_base_2d_inline1077__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v137 = pto::Shape<1, 1, 1, -1, -1>(v132, v133, v134, v65, v66);
  // pto: %hc_base_2d_inline1077__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v138 = pto::Stride<-1, -1, -1, -1, -1>(v133 * v136, v136, v135, v66, v65);
  // pto: %hc_base_2d_inline1077__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v139 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v8, v137, v138);
  // pto: %comb_inline1100__ssa_v0_view
  const int64_t v140 = 1;
  // pto: %comb_inline1100__ssa_v0_view
  const int64_t v141 = 1;
  // pto: %comb_inline1100__ssa_v0_view
  const int64_t v142 = 1;
  // pto: %comb_inline1100__ssa_v0_view
  int64_t v143 = (int64_t) v29;
  // pto: %comb_inline1100__ssa_v0_view
  int64_t v144 = v143 * v70;
  // pto: %comb_inline1100__ssa_v0_view
  int64_t v145 = v142 * v144;
  // pto: %comb_inline1100__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v146 = pto::Shape<1, 1, 1, -1, -1>(v140, v141, v142, v143, v70);
  // pto: %comb_inline1100__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v147 = pto::Stride<-1, -1, -1, -1, -1>(v141 * v145, v145, v144, v70, v65);
  // pto: %comb_inline1100__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v148 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v9, v146, v147);
  // pto: %hc_scale_2d_inline1168__ssa_v0_view
  const int64_t v149 = 1;
  // pto: %hc_scale_2d_inline1168__ssa_v0_view
  const int64_t v150 = 1;
  // pto: %hc_scale_2d_inline1168__ssa_v0_view
  const int64_t v151 = 1;
  // pto: %hc_scale_2d_inline1168__ssa_v0_view
  int64_t v152 = v65 * v68;
  // pto: %hc_scale_2d_inline1168__ssa_v0_view
  int64_t v153 = v151 * v152;
  // pto: %hc_scale_2d_inline1168__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v154 = pto::Shape<1, 1, 1, -1, -1>(v149, v150, v151, v65, v68);
  // pto: %hc_scale_2d_inline1168__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v155 = pto::Stride<-1, -1, -1, -1, -1>(v150 * v153, v153, v152, v68, v65);
  // pto: %hc_scale_2d_inline1168__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v156 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v10, v154, v155);
  // pto: %225
  int64_t v157 = (int64_t) ((uint64_t) v29 * (uint64_t) v77);
  // pto: %post_flat_inline1112__ssa_v0_view
  const int64_t v158 = 1;
  // pto: %post_flat_inline1112__ssa_v0_view
  const int64_t v159 = 1;
  // pto: %post_flat_inline1112__ssa_v0_view
  const int64_t v160 = 1;
  // pto: %post_flat_inline1112__ssa_v0_view
  int64_t v161 = v65 * v157;
  // pto: %post_flat_inline1112__ssa_v0_view
  int64_t v162 = v160 * v161;
  // pto: %post_flat_inline1112__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v163 = pto::Shape<1, 1, 1, -1, -1>(v158, v159, v160, v65, v157);
  // pto: %post_flat_inline1112__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v164 = pto::Stride<-1, -1, -1, -1, -1>(v159 * v162, v162, v161, v157, v65);
  // pto: %post_flat_inline1112__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v165 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v11, v163, v164);
  // pto: %mixed_view_inline1121__ssa_v0_view
  const int64_t v166 = 1;
  // pto: %mixed_view_inline1121__ssa_v0_view
  const int64_t v167 = 1;
  // pto: %mixed_view_inline1121__ssa_v0_view
  const int64_t v168 = 1;
  // pto: %mixed_view_inline1121__ssa_v0_view
  int64_t v169 = (int64_t) v29;
  // pto: %mixed_view_inline1121__ssa_v0_view
  int64_t v170 = v169 * v71;
  // pto: %mixed_view_inline1121__ssa_v0_view
  int64_t v171 = v168 * v170;
  // pto: %mixed_view_inline1121__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v172 = pto::Shape<1, 1, 1, -1, -1>(v166, v167, v168, v169, v71);
  // pto: %mixed_view_inline1121__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v173 = pto::Stride<-1, -1, -1, -1, -1>(v167 * v171, v171, v170, v71, v65);
  // pto: %mixed_view_inline1121__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v174 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v12, v172, v173);
  const int32_t v175 = 0;
  auto v176 = TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>(v13, v175, v72);
  // pto: %worker_inline1090__ssa_v0
  int64_t v177 = (int64_t) v30;
  // pto: %linear_workers_inline1120__ssa_v0_idx
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  set_flag(PIPE_S, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  for (int64_t v178 = v177; v178 < v14; v178 += (int64_t) v15) {
    // pto: %34
    ;
    int64_t v179 = v178 % v69;
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    for (int64_t v180 = v74; v180 < v75; v180 += v65) {
      // pto: %35, %37, %36
      ;
      int64_t v181 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v179 * (uint64_t) v73) + (uint64_t) ((int64_t) (uint64_t) v180 * (uint64_t) v76));
      // pto: %32, %33, %39, %lane_inline1050__ssa_v0, %38
      ;
      int64_t v182 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) (v178 / v69) * (uint64_t) v64) + (uint64_t) ((int64_t) (uint64_t) ((int64_t) v32) * (uint64_t) v70));
      // pto: %stream_source_inline1049__ssa_v0
      ;
      Tile<TileType::Vec, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v183 = Tile<TileType::Vec, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
      // pto: %stream_source_inline1049__ssa_v0
      ;
      uint64_t v184 = (uint64_t) v33;
      TASSIGN(v183, v184);
      // pto: %40
      ;
      int64_t v185 = v182 < v74 ? v74 : v182;
      // pto: %41
      ;
      int64_t v186 = v181 < v74 ? v74 : v181;
      // pto: %x_flat_inline1075__ssa_v0_pview
      ;
      __gm__ bfloat16_t* v187 = PTOAS__GLOBAL_TENSOR_DATA(v113);
      // pto: %x_flat_inline1075__ssa_v0_pview
      ;
      const int64_t v188 = 0;
      // pto: %x_flat_inline1075__ssa_v0_pview
      ;
      const int64_t v189 = 16384;
      // pto: %x_flat_inline1075__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 16, 1024> v190 = pto::Shape<1, 1, 1, 16, 1024>();
      // pto: %x_flat_inline1075__ssa_v0_pview
      ;
      pto::Stride<262144, 262144, 262144, 16384, 1> v191 = pto::Stride<262144, 262144, 262144, 16384, 1>();
      // pto: %x_flat_inline1075__ssa_v0_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<262144, 262144, 262144, 16384, 1>, pto::Layout::ND> v192 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<262144, 262144, 262144, 16384, 1>, pto::Layout::ND>(v187 + (v188 + v185 * v189 + v186), v190, v191);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v183, v192);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %stream_widened_inline1047__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v193 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
      // pto: %stream_widened_inline1047__ssa_v0
      ;
      uint64_t v194 = (uint64_t) v34;
      TASSIGN(v193, v194);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      RoundMode v195 = RoundMode::CAST_ROUND;
      SaturationMode v196 = SaturationMode::ON;
      TCVT(v193, v183, v195, v196);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TPUSH<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v176, v193);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      // pto: %stream_reduce_tmp_inline1044__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v197 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
      // pto: %stream_reduce_tmp_inline1044__ssa_v0
      ;
      uint64_t v198 = (uint64_t) v33;
      TASSIGN(v197, v198);
      // pto: %t__tmp_v233
      ;
      Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v199 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v76);
      // pto: %t__tmp_v233
      ;
      uint64_t v200 = (uint64_t) v34;
      TASSIGN(v199, v200);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      TMUL(v199, v193, v193);
      // pto: %stream_sq_inline1019__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v201 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v65);
      // pto: %stream_sq_inline1019__ssa_v0
      ;
      uint64_t v202 = (uint64_t) v35;
      TASSIGN(v201, v202);
      pipe_barrier(PIPE_V);
      TROWSUM(v201, v199, v197);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      // pto: %42, %43, %44, %45
      ;
      int64_t v203 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v179 * (uint64_t) v75) + (uint64_t) v180) * (uint64_t) v16) + (uint64_t) v182);
      // pto: %46
      ;
      int64_t v204 = v203 < v74 ? v74 : v203;
      // pto: %rms_partials_inline1178__iter_v1_pview
      ;
      const int64_t v205 = 0;
      // pto: %rms_partials_inline1178__iter_v1_pview
      ;
      __gm__ float* v206 = PTOAS__GLOBAL_TENSOR_DATA(v104);
      // pto: %rms_partials_inline1178__iter_v1_pview
      ;
      const int64_t v207 = 1;
      // pto: %rms_partials_inline1178__iter_v1_pview
      ;
      const int64_t v208 = 1;
      // pto: %rms_partials_inline1178__iter_v1_pview
      ;
      const int64_t v209 = 1;
      // pto: %rms_partials_inline1178__iter_v1_pview
      ;
      int64_t v210 = v70 * v65;
      // pto: %rms_partials_inline1178__iter_v1_pview
      ;
      int64_t v211 = v209 * v210;
      // pto: %rms_partials_inline1178__iter_v1_pview
      ;
      pto::Shape<1, 1, 1, 16, 1> v212 = pto::Shape<1, 1, 1, 16, 1>(v207, v208, v209, v70, v65);
      // pto: %rms_partials_inline1178__iter_v1_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v213 = pto::Stride<-1, -1, -1, -1, -1>(v208 * v211, v211, v210, v65, v96);
      // pto: %rms_partials_inline1178__iter_v1_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v214 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v206 + (v205 + v204 * v65 + v74 * v96), v212, v213);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      TSTORE(v214, v201);
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
  // pto: %47, %48, %tail_lane_inline1001__ssa_v0
  int64_t v215 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v177 * (uint64_t) v75) + (uint64_t) ((int64_t) v32));
  // pto: %49
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  if (v215 < v17) {
    // pto: %51
    ;
    int64_t v216 = v215 < v19 ? v215 : v19;
    // pto: %50, %52
    ;
    int64_t v217 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v215 * (uint64_t) v18) + (uint64_t) v216);
    // pto: %54
    ;
    int64_t v218 = (int64_t) ((uint64_t) v19 - (uint64_t) v215);
    // pto: %55
    ;
    int64_t v219 = v218 < v74 ? v74 : v218;
    // pto: %56
    ;
    int64_t v220 = v219 < v65 ? v219 : v65;
    // pto: %53, %57
    ;
    int64_t v221 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v217 + (uint64_t) v18) + (uint64_t) v220);
    for (int64_t v222 = v217; v222 < v221; v222 += v77) {
      // pto: %scale2_inline1114__tile
      ;
      float v223 = (v5)[v75];
      // pto: %58
      ;
      int64_t v224 = (int64_t) ((uint64_t) v221 - (uint64_t) v222);
      // pto: %59
      ;
      int64_t v225 = v224 < v77 ? v224 : v77;
      // pto: %60
      ;
      int64_t v226 = (int64_t) ((uint64_t) v225 * (uint64_t) v64);
      // pto: %c_parts_load_inline997__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v227 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v226);
      // pto: %c_parts_load_inline997__ssa_v0
      ;
      uint64_t v228 = (uint64_t) v34;
      TASSIGN(v227, v228);
      // pto: %61
      ;
      int64_t v229 = (int64_t) ((uint64_t) v222 * (uint64_t) v64);
      // pto: %62
      ;
      int64_t v230 = v229 < v74 ? v74 : v229;
      // pto: %parts_2d_inline1150__ssa_v0_pview
      ;
      const int64_t v231 = 0;
      // pto: %parts_2d_inline1150__ssa_v0_pview
      ;
      __gm__ float* v232 = PTOAS__GLOBAL_TENSOR_DATA(v122);
      // pto: %parts_2d_inline1150__ssa_v0_pview
      ;
      const int64_t v233 = 1;
      // pto: %parts_2d_inline1150__ssa_v0_pview
      ;
      const int64_t v234 = 1;
      // pto: %parts_2d_inline1150__ssa_v0_pview
      ;
      const int64_t v235 = 1;
      // pto: %parts_2d_inline1150__ssa_v0_pview
      ;
      int64_t v236 = v69 * v114;
      // pto: %parts_2d_inline1150__ssa_v0_pview
      ;
      int64_t v237 = v235 * v236;
      // pto: %parts_2d_inline1150__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, -1> v238 = pto::Shape<1, 1, 1, 8, -1>(v233, v234, v235, v69, v226);
      // pto: %parts_2d_inline1150__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v239 = pto::Stride<-1, -1, -1, -1, -1>(v234 * v237, v237, v236, v114, v65);
      // pto: %parts_2d_inline1150__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v240 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v232 + (v231 + v74 * v114 + v230 * v65), v238, v239);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      TLOAD(v227, v240);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %t__tmp_v234
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v241 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v69, v78);
      // pto: %t__tmp_v234
      ;
      uint64_t v242 = (uint64_t) v34;
      TASSIGN(v241, v242);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      TFILLPAD<pto::TFillPadMode::InPlace>(v241, v227);
      v241.SetValidShape(v69, v78);
      // pto: %c_parts_tmp_inline1129__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v243 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v78);
      // pto: %c_parts_tmp_inline1129__ssa_v0
      ;
      uint64_t v244 = (uint64_t) v36;
      TASSIGN(v243, v244);
      // pto: %t__tmp_v235
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v245 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v78);
      // pto: %t__tmp_v235
      ;
      uint64_t v246 = (uint64_t) v34;
      TASSIGN(v245, v246);
      // pto: %colsum_src_view
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, 8, 256, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v247;
      TRESHAPE(v247, v241);
      // pto: %colsum_tmp_view
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, 8, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v248;
      TRESHAPE(v248, v243);
      // pto: %colsum_dst_view
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v249;
      TRESHAPE(v249, v245);
      pipe_barrier(PIPE_V);
      const bool v250 = true;
      TCOLSUM(v249, v247, v248, v250);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
      // pto: %c_total_inline996__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v251 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v64);
      // pto: %c_total_inline996__ssa_v0
      ;
      uint64_t v252 = (uint64_t) v34;
      TASSIGN(v251, v252);
      // pto: %c_sq_load_inline995__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v253 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v225);
      // pto: %c_sq_load_inline995__ssa_v0
      ;
      uint64_t v254 = (uint64_t) v37;
      TASSIGN(v253, v254);
      // pto: %64
      ;
      int64_t v255 = v222 < v74 ? v74 : v222;
      // pto: %rms_2d_inline1060__ssa_v0_pview
      ;
      int64_t v256 = (int64_t) v28;
      // pto: %rms_2d_inline1060__ssa_v0_pview
      ;
      const int64_t v257 = 0;
      // pto: %rms_2d_inline1060__ssa_v0_pview
      ;
      __gm__ float* v258 = PTOAS__GLOBAL_TENSOR_DATA(v131);
      // pto: %rms_2d_inline1060__ssa_v0_pview
      ;
      const int64_t v259 = 1;
      // pto: %rms_2d_inline1060__ssa_v0_pview
      ;
      const int64_t v260 = 1;
      // pto: %rms_2d_inline1060__ssa_v0_pview
      ;
      const int64_t v261 = 1;
      // pto: %rms_2d_inline1060__ssa_v0_pview
      ;
      int64_t v262 = v70 * v256;
      // pto: %rms_2d_inline1060__ssa_v0_pview
      ;
      int64_t v263 = v261 * v262;
      // pto: %rms_2d_inline1060__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 16, -1> v264 = pto::Shape<1, 1, 1, 16, -1>(v259, v260, v261, v70, v225);
      // pto: %rms_2d_inline1060__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v265 = pto::Stride<-1, -1, -1, -1, -1>(v260 * v263, v263, v262, v256, v65);
      // pto: %rms_2d_inline1060__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v266 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v258 + (v257 + v74 * v256 + v255 * v65), v264, v265);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
      TLOAD(v253, v266);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %t__tmp_v236
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v267 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v70, v69);
      // pto: %t__tmp_v236
      ;
      uint64_t v268 = (uint64_t) v37;
      TASSIGN(v267, v268);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      TFILLPAD<pto::TFillPadMode::InPlace>(v267, v253);
      v267.SetValidShape(v70, v69);
      // pto: %c_sq_tmp_inline1212__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v269 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v69);
      // pto: %c_sq_tmp_inline1212__ssa_v0
      ;
      uint64_t v270 = (uint64_t) v38;
      TASSIGN(v269, v270);
      // pto: %c_rms_tmp_inline1014__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v271 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %c_rms_tmp_inline1014__ssa_v0
      ;
      uint64_t v272 = (uint64_t) v39;
      TASSIGN(v271, v272);
      // pto: %t__tmp_v237
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v273 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v237
      ;
      uint64_t v274 = (uint64_t) v37;
      TASSIGN(v273, v274);
      // pto: %65
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, 16, 8, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v275;
      TRESHAPE(v275, v267);
      // pto: %66
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, 16, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v276;
      TRESHAPE(v276, v269);
      // pto: %67
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v277;
      TRESHAPE(v277, v273);
      pipe_barrier(PIPE_V);
      const bool v278 = true;
      TCOLSUM(v277, v275, v276, v278);
      // pto: %t__tmp_v238
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v279 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v238
      ;
      uint64_t v280 = (uint64_t) v37;
      TASSIGN(v279, v280);
      pipe_barrier(PIPE_V);
      TMULS(v279, v273, v79);
      // pto: %t__tmp_v239
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v281 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v239
      ;
      uint64_t v282 = (uint64_t) v37;
      TASSIGN(v281, v282);
      pipe_barrier(PIPE_V);
      TADDS(v281, v279, v80);
      // pto: %t__tmp_v240
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v283 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v240
      ;
      uint64_t v284 = (uint64_t) v40;
      TASSIGN(v283, v284);
      pipe_barrier(PIPE_V);
      TRSQRT(v283, v281, v271);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      // pto: %c_inv_inline1036__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v285 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v65);
      // pto: %c_inv_inline1036__ssa_v0
      ;
      uint64_t v286 = (uint64_t) v40;
      TASSIGN(v285, v286);
      // pto: %comb_inv_inline1170__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v287 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v65);
      // pto: %comb_inv_inline1170__ssa_v0
      ;
      uint64_t v288 = (uint64_t) v40;
      TASSIGN(v287, v288);
      // pto: %t__tmp_v241
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v289 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v70);
      // pto: %t__tmp_v241
      ;
      uint64_t v290 = (uint64_t) v41;
      TASSIGN(v289, v290);
      // pto: %slice_view
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 16, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v291;
      // pto: %slice_view
      ;
      uint64_t v292 = (uint64_t) v41;
      TASSIGN(v291, v292);
      // pto: %raw_inline1172__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v293 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v70);
      // pto: %raw_inline1172__ssa_v0
      ;
      uint64_t v294 = (uint64_t) v34;
      TASSIGN(v293, v294);
      TADDS(v293, v291, v81);
      // pto: %base_inline1173__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v295 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v70);
      // pto: %base_inline1173__ssa_v0
      ;
      uint64_t v296 = (uint64_t) v37;
      TASSIGN(v295, v296);
      // pto: %hc_base_2d_inline1077__ssa_v0_pview
      ;
      __gm__ float* v297 = PTOAS__GLOBAL_TENSOR_DATA(v139);
      // pto: %hc_base_2d_inline1077__ssa_v0_pview
      ;
      unsigned v298 = 8;
      // pto: %hc_base_2d_inline1077__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 16> v299 = pto::Shape<1, 1, 1, 1, 16>();
      // pto: %hc_base_2d_inline1077__ssa_v0_pview
      ;
      pto::Stride<24, 24, 24, 24, 1> v300 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %hc_base_2d_inline1077__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v301 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v297 + v298, v299, v300);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      TLOAD(v295, v301);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      // pto: %t__tmp_v242
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v302 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v70);
      // pto: %t__tmp_v242
      ;
      uint64_t v303 = (uint64_t) v42;
      TASSIGN(v302, v303);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v302, v293, v287);
      // pto: %t__tmp_v243
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v304 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v70);
      // pto: %t__tmp_v243
      ;
      uint64_t v305 = (uint64_t) v42;
      TASSIGN(v304, v305);
      pipe_barrier(PIPE_V);
      TMULS(v304, v302, v223);
      // pto: %t__tmp_v244
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v306 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v70);
      // pto: %t__tmp_v244
      ;
      uint64_t v307 = (uint64_t) v34;
      TASSIGN(v306, v307);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      TCOLEXPAND(v306, v295);
      // pto: %logits_inline1058__ssa_v1
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v308 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v70);
      // pto: %logits_inline1058__ssa_v1
      ;
      uint64_t v309 = (uint64_t) v43;
      TASSIGN(v308, v309);
      pipe_barrier(PIPE_V);
      TADD(v308, v304, v306);
      // pto: %t__tmp_v245
      ;
      Tile<TileType::Vec, float, 4, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v310 = Tile<TileType::Vec, float, 4, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v77, v70);
      // pto: %t__tmp_v245
      ;
      uint64_t v311 = (uint64_t) v43;
      TASSIGN(v310, v311);
      // pto: %logits_flat_inline1174__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v312 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %logits_flat_inline1174__ssa_v0
      ;
      uint64_t v313 = (uint64_t) v43;
      TASSIGN(v312, v313);
      // pto: %ids_inline1175__ci_tmp_v0
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v314 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v83);
      // pto: %ids_inline1175__ci_tmp_v0
      ;
      uint64_t v315 = (uint64_t) v34;
      TASSIGN(v314, v315);
      // pto: %ids_inline1175__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v316 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %ids_inline1175__ssa_v0
      ;
      uint64_t v317 = (uint64_t) v44;
      TASSIGN(v316, v317);
      // pto: %ci_tmp_view
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v318;
      TRESHAPE(v318, v314);
      // pto: %ci_dst_view
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v319;
      TRESHAPE(v319, v316);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v319, v72, v318);
      // pto: %tmp_inline1177__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v320 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %tmp_inline1177__ssa_v0
      ;
      uint64_t v321 = (uint64_t) v38;
      TASSIGN(v320, v321);
      // pto: %component_ids_inline1182__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v322 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %component_ids_inline1182__ssa_v0
      ;
      uint64_t v323 = (uint64_t) v34;
      TASSIGN(v322, v323);
      pipe_barrier(PIPE_V);
      TSHRS(v322, v316, v84);
      // pto: %t__tmp_v246
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v324 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v246
      ;
      uint64_t v325 = (uint64_t) v45;
      TASSIGN(v324, v325);
      pipe_barrier(PIPE_V);
      TSHLS(v324, v322, v84);
      // pto: %token_ids_inline1183__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v326 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %token_ids_inline1183__ssa_v0
      ;
      uint64_t v327 = (uint64_t) v45;
      TASSIGN(v326, v327);
      pipe_barrier(PIPE_V);
      TSUB(v326, v316, v324);
      // pto: %t__tmp_v247
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v328 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v247
      ;
      uint64_t v329 = (uint64_t) v45;
      TASSIGN(v328, v329);
      pipe_barrier(PIPE_V);
      TSHLS(v328, v326, v85);
      // pto: %perm_in_inline1187__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v330 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %perm_in_inline1187__ssa_v0
      ;
      uint64_t v331 = (uint64_t) v34;
      TASSIGN(v330, v331);
      pipe_barrier(PIPE_V);
      TADD(v330, v328, v322);
      // pto: %out_token_inline1108__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v332 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %out_token_inline1108__ssa_v0
      ;
      uint64_t v333 = (uint64_t) v45;
      TASSIGN(v332, v333);
      pipe_barrier(PIPE_V);
      TSHRS(v332, v316, v85);
      // pto: %t__tmp_v248
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v334 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v248
      ;
      uint64_t v335 = (uint64_t) v42;
      TASSIGN(v334, v335);
      pipe_barrier(PIPE_V);
      TSHLS(v334, v332, v85);
      // pto: %out_component_inline1188__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v336 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %out_component_inline1188__ssa_v0
      ;
      uint64_t v337 = (uint64_t) v42;
      TASSIGN(v336, v337);
      pipe_barrier(PIPE_V);
      TSUB(v336, v316, v334);
      // pto: %t__tmp_v249
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v338 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v249
      ;
      uint64_t v339 = (uint64_t) v42;
      TASSIGN(v338, v339);
      pipe_barrier(PIPE_V);
      TSHLS(v338, v336, v84);
      // pto: %perm_out_inline1171__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v340 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %perm_out_inline1171__ssa_v0
      ;
      uint64_t v341 = (uint64_t) v46;
      TASSIGN(v340, v341);
      pipe_barrier(PIPE_V);
      TADD(v340, v338, v332);
      // pto: %compact_logits_inline1190__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v342 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %compact_logits_inline1190__ssa_v0
      ;
      uint64_t v343 = (uint64_t) v39;
      TASSIGN(v342, v343);
      TGATHER(v342, v312, v330, v320);
      // pto: %ids16_inline1105__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v344 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v86);
      // pto: %ids16_inline1105__ssa_v0
      ;
      uint64_t v345 = (uint64_t) v44;
      TASSIGN(v344, v345);
      // pto: %xor_tmp16_inline1278__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v346 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v86);
      // pto: %xor_tmp16_inline1278__ssa_v0
      ;
      uint64_t v347 = (uint64_t) v43;
      TASSIGN(v346, v347);
      // pto: %block_numbers_inline1099__ci_tmp_v1
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v348 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v83);
      // pto: %block_numbers_inline1099__ci_tmp_v1
      ;
      uint64_t v349 = (uint64_t) v34;
      TASSIGN(v348, v349);
      // pto: %block_numbers_inline1099__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v350 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %block_numbers_inline1099__ssa_v0
      ;
      uint64_t v351 = (uint64_t) v37;
      TASSIGN(v350, v351);
      // pto: %69
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v352;
      TRESHAPE(v352, v348);
      // pto: %70
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v353;
      TRESHAPE(v353, v350);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v353, v72, v352);
      // pto: %block_ids_inline1192__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v354 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %block_ids_inline1192__ssa_v0
      ;
      uint64_t v355 = (uint64_t) v42;
      TASSIGN(v354, v355);
      pipe_barrier(PIPE_V);
      TSHLS(v354, v350, v87);
      // pto: %block_ids16_inline1193__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v356 = Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v70);
      // pto: %block_ids16_inline1193__ssa_v0
      ;
      uint64_t v357 = (uint64_t) v42;
      TASSIGN(v356, v357);
      // pto: %block_tmp16_inline1195__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v358 = Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v70);
      // pto: %block_tmp16_inline1195__ssa_v0
      ;
      uint64_t v359 = (uint64_t) v47;
      TASSIGN(v358, v359);
      // pto: %t__tmp_v250
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v360 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v250
      ;
      uint64_t v361 = (uint64_t) v34;
      TASSIGN(v360, v361);
      TEXPANDS(v360, v85);
      // pto: %ri1_mask_inline1126__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v362 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v86);
      // pto: %ri1_mask_inline1126__ssa_v0
      ;
      uint64_t v363 = (uint64_t) v34;
      TASSIGN(v362, v363);
      // pto: %t__tmp_v251
      ;
      Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v364 = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v86);
      // pto: %t__tmp_v251
      ;
      uint64_t v365 = (uint64_t) v45;
      TASSIGN(v364, v365);
      pipe_barrier(PIPE_V);
      TXOR(v364, v344, v362, v346);
      // pto: %ri1_inline1196__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v366 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %ri1_inline1196__ssa_v0
      ;
      uint64_t v367 = (uint64_t) v45;
      TASSIGN(v366, v367);
      // pto: %t__tmp_v252
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v368 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v252
      ;
      uint64_t v369 = (uint64_t) v34;
      TASSIGN(v368, v369);
      pipe_barrier(PIPE_V);
      TEXPANDS(v368, v88);
      // pto: %ri2_block_mask_inline1197__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v370 = Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v70);
      // pto: %ri2_block_mask_inline1197__ssa_v0
      ;
      uint64_t v371 = (uint64_t) v34;
      TASSIGN(v370, v371);
      // pto: %t__tmp_v253
      ;
      Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v372 = Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v70);
      // pto: %t__tmp_v253
      ;
      uint64_t v373 = (uint64_t) v44;
      TASSIGN(v372, v373);
      pipe_barrier(PIPE_V);
      TXOR(v372, v356, v370, v358);
      // pto: %ri2_offset_inline1198__ssa_v0
      ;
      Tile<TileType::Vec, uint32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v374 = Tile<TileType::Vec, uint32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %ri2_offset_inline1198__ssa_v0
      ;
      uint64_t v375 = (uint64_t) v44;
      TASSIGN(v374, v375);
      // pto: %t__tmp_v254
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v376 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v254
      ;
      uint64_t v377 = (uint64_t) v34;
      TASSIGN(v376, v377);
      pipe_barrier(PIPE_V);
      TEXPANDS(v376, v89);
      // pto: %ci1_block_mask_inline1110__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v378 = Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v70);
      // pto: %ci1_block_mask_inline1110__ssa_v0
      ;
      uint64_t v379 = (uint64_t) v34;
      TASSIGN(v378, v379);
      // pto: %t__tmp_v255
      ;
      Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v380 = Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v70);
      // pto: %t__tmp_v255
      ;
      uint64_t v381 = (uint64_t) v48;
      TASSIGN(v380, v381);
      pipe_barrier(PIPE_V);
      TXOR(v380, v356, v378, v358);
      // pto: %ci1_offset_inline1201__ssa_v0
      ;
      Tile<TileType::Vec, uint32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v382 = Tile<TileType::Vec, uint32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %ci1_offset_inline1201__ssa_v0
      ;
      uint64_t v383 = (uint64_t) v48;
      TASSIGN(v382, v383);
      // pto: %t__tmp_v256
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v384 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v256
      ;
      uint64_t v385 = (uint64_t) v34;
      TASSIGN(v384, v385);
      pipe_barrier(PIPE_V);
      TEXPANDS(v384, v90);
      // pto: %ci2_block_mask_inline1013__ssa_v0
      ;
      Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v386 = Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v70);
      // pto: %ci2_block_mask_inline1013__ssa_v0
      ;
      uint64_t v387 = (uint64_t) v34;
      TASSIGN(v386, v387);
      // pto: %t__tmp_v257
      ;
      Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v388 = Tile<TileType::Vec, int16_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v70);
      // pto: %t__tmp_v257
      ;
      uint64_t v389 = (uint64_t) v49;
      TASSIGN(v388, v389);
      pipe_barrier(PIPE_V);
      TXOR(v388, v356, v386, v358);
      // pto: %ci2_offset_inline1285__ssa_v0
      ;
      Tile<TileType::Vec, uint32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v390 = Tile<TileType::Vec, uint32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %ci2_offset_inline1285__ssa_v0
      ;
      uint64_t v391 = (uint64_t) v49;
      TASSIGN(v390, v391);
      // pto: %sink_swap_0_inline1202__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v392 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %sink_swap_0_inline1202__ssa_v0
      ;
      uint64_t v393 = (uint64_t) v34;
      TASSIGN(v392, v393);
      pipe_barrier(PIPE_V);
      TGATHER(v392, v342, v366, v320);
      // pto: %max2_inline1235__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v394 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %max2_inline1235__ssa_v0
      ;
      uint64_t v395 = (uint64_t) v34;
      TASSIGN(v394, v395);
      pipe_barrier(PIPE_V);
      TMAX(v394, v342, v392);
      // pto: %t__tmp_v258
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v396 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v258
      ;
      uint64_t v397 = (uint64_t) v42;
      TASSIGN(v396, v397);
      pipe_barrier(PIPE_V);
      TGATHERB(v396, v394, v374);
      // pto: %sink_swap_1_inline1206__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v398 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %sink_swap_1_inline1206__ssa_v0
      ;
      uint64_t v399 = (uint64_t) v42;
      TASSIGN(v398, v399);
      // pto: %71
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v400;
      // pto: %71
      ;
      uint64_t v401 = (uint64_t) v42;
      TASSIGN(v400, v401);
      // pto: %max4_inline1208__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v402 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %max4_inline1208__ssa_v0
      ;
      uint64_t v403 = (uint64_t) v34;
      TASSIGN(v402, v403);
      pipe_barrier(PIPE_V);
      TMAX(v402, v394, v400);
      // pto: %t__tmp_v259
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v404 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v259
      ;
      uint64_t v405 = (uint64_t) v34;
      TASSIGN(v404, v405);
      pipe_barrier(PIPE_V);
      TSUB(v404, v342, v402);
      // pto: %exps_inline1252__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v406 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %exps_inline1252__ssa_v0
      ;
      uint64_t v407 = (uint64_t) v34;
      TASSIGN(v406, v407);
      pipe_barrier(PIPE_V);
      TEXP(v406, v404);
      // pto: %sink_swap_2_inline1209__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v408 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %sink_swap_2_inline1209__ssa_v0
      ;
      uint64_t v409 = (uint64_t) v42;
      TASSIGN(v408, v409);
      pipe_barrier(PIPE_V);
      TGATHER(v408, v406, v366, v320);
      // pto: %sum2_inline1210__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v410 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %sum2_inline1210__ssa_v0
      ;
      uint64_t v411 = (uint64_t) v42;
      TASSIGN(v410, v411);
      pipe_barrier(PIPE_V);
      TADD(v410, v406, v408);
      // pto: %t__tmp_v260
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v412 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v260
      ;
      uint64_t v413 = (uint64_t) v43;
      TASSIGN(v412, v413);
      pipe_barrier(PIPE_V);
      TGATHERB(v412, v410, v374);
      // pto: %sink_swap_3_inline1251__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v414 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %sink_swap_3_inline1251__ssa_v0
      ;
      uint64_t v415 = (uint64_t) v43;
      TASSIGN(v414, v415);
      // pto: %72
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v416;
      // pto: %72
      ;
      uint64_t v417 = (uint64_t) v43;
      TASSIGN(v416, v417);
      // pto: %sum4_inline1154__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v418 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %sum4_inline1154__ssa_v0
      ;
      uint64_t v419 = (uint64_t) v42;
      TASSIGN(v418, v419);
      pipe_barrier(PIPE_V);
      TADD(v418, v410, v416);
      // pto: %t__tmp_v261
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v420 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v261
      ;
      uint64_t v421 = (uint64_t) v34;
      TASSIGN(v420, v421);
      pipe_barrier(PIPE_V);
      TDIV(v420, v406, v418);
      // pto: %probs_inline1119__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v422 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %probs_inline1119__ssa_v0
      ;
      uint64_t v423 = (uint64_t) v34;
      TASSIGN(v422, v423);
      pipe_barrier(PIPE_V);
      TADDS(v422, v420, v80);
      // pto: %t__tmp_v262
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v424 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v262
      ;
      uint64_t v425 = (uint64_t) v42;
      TASSIGN(v424, v425);
      pipe_barrier(PIPE_V);
      TGATHERB(v424, v422, v382);
      // pto: %sink_swap_4_inline1179__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v426 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %sink_swap_4_inline1179__ssa_v0
      ;
      uint64_t v427 = (uint64_t) v42;
      TASSIGN(v426, v427);
      // pto: %73
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v428;
      // pto: %73
      ;
      uint64_t v429 = (uint64_t) v42;
      TASSIGN(v428, v429);
      // pto: %col2_inline1211__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v430 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %col2_inline1211__ssa_v0
      ;
      uint64_t v431 = (uint64_t) v42;
      TASSIGN(v430, v431);
      pipe_barrier(PIPE_V);
      TADD(v430, v422, v428);
      // pto: %t__tmp_v263
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v432 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v263
      ;
      uint64_t v433 = (uint64_t) v43;
      TASSIGN(v432, v433);
      pipe_barrier(PIPE_V);
      TGATHERB(v432, v430, v390);
      // pto: %sink_swap_5_inline1138__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v434 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %sink_swap_5_inline1138__ssa_v0
      ;
      uint64_t v435 = (uint64_t) v43;
      TASSIGN(v434, v435);
      // pto: %74
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v436;
      // pto: %74
      ;
      uint64_t v437 = (uint64_t) v43;
      TASSIGN(v436, v437);
      // pto: %t__tmp_v264
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v438 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v264
      ;
      uint64_t v439 = (uint64_t) v42;
      TASSIGN(v438, v439);
      pipe_barrier(PIPE_V);
      TADD(v438, v430, v436);
      // pto: %col4_inline1213__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v440 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %col4_inline1213__ssa_v0
      ;
      uint64_t v441 = (uint64_t) v42;
      TASSIGN(v440, v441);
      pipe_barrier(PIPE_V);
      TADDS(v440, v438, v80);
      // pto: %cur_inline1151__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v442 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %cur_inline1151__ssa_v0
      ;
      uint64_t v443 = (uint64_t) v34;
      TASSIGN(v442, v443);
      pipe_barrier(PIPE_V);
      TDIV(v442, v422, v440);
      for (int64_t v444 = v74; v444 < v91; v444 += v65) {
        // pto: %sink_swap_6_inline1176__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v445 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %sink_swap_6_inline1176__ssa_v0
        ;
        uint64_t v446 = (uint64_t) v42;
        TASSIGN(v445, v446);
        pipe_barrier(PIPE_V);
        TGATHER(v445, v442, v366, v320);
        // pto: %row2_inline1228__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v447 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %row2_inline1228__ssa_v0
        ;
        uint64_t v448 = (uint64_t) v42;
        TASSIGN(v447, v448);
        pipe_barrier(PIPE_V);
        TADD(v447, v442, v445);
        // pto: %t__tmp_v265
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v449 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %t__tmp_v265
        ;
        uint64_t v450 = (uint64_t) v43;
        TASSIGN(v449, v450);
        pipe_barrier(PIPE_V);
        TGATHERB(v449, v447, v374);
        // pto: %sink_swap_7_inline1218__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v451 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %sink_swap_7_inline1218__ssa_v0
        ;
        uint64_t v452 = (uint64_t) v43;
        TASSIGN(v451, v452);
        // pto: %75
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v453;
        // pto: %75
        ;
        uint64_t v454 = (uint64_t) v43;
        TASSIGN(v453, v454);
        // pto: %t__tmp_v266
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v455 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %t__tmp_v266
        ;
        uint64_t v456 = (uint64_t) v42;
        TASSIGN(v455, v456);
        pipe_barrier(PIPE_V);
        TADD(v455, v447, v453);
        // pto: %row4_inline1087__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v457 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %row4_inline1087__ssa_v0
        ;
        uint64_t v458 = (uint64_t) v42;
        TASSIGN(v457, v458);
        pipe_barrier(PIPE_V);
        TADDS(v457, v455, v80);
        // pto: %norm_inline1220__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v459 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %norm_inline1220__ssa_v0
        ;
        uint64_t v460 = (uint64_t) v42;
        TASSIGN(v459, v460);
        pipe_barrier(PIPE_V);
        TDIV(v459, v442, v457);
        // pto: %t__tmp_v267
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v461 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %t__tmp_v267
        ;
        uint64_t v462 = (uint64_t) v43;
        TASSIGN(v461, v462);
        pipe_barrier(PIPE_V);
        TGATHERB(v461, v459, v382);
        // pto: %sink_swap_8_inline1230__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v463 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %sink_swap_8_inline1230__ssa_v0
        ;
        uint64_t v464 = (uint64_t) v43;
        TASSIGN(v463, v464);
        // pto: %76
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v465;
        // pto: %76
        ;
        uint64_t v466 = (uint64_t) v43;
        TASSIGN(v465, v466);
        // pto: %col2_inline1211__ssa_v3
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v467 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %col2_inline1211__ssa_v3
        ;
        uint64_t v468 = (uint64_t) v43;
        TASSIGN(v467, v468);
        pipe_barrier(PIPE_V);
        TADD(v467, v459, v465);
        // pto: %t__tmp_v268
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v469 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %t__tmp_v268
        ;
        uint64_t v470 = (uint64_t) v37;
        TASSIGN(v469, v470);
        pipe_barrier(PIPE_V);
        TGATHERB(v469, v467, v390);
        // pto: %sink_swap_9_inline1221__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v471 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %sink_swap_9_inline1221__ssa_v0
        ;
        uint64_t v472 = (uint64_t) v37;
        TASSIGN(v471, v472);
        // pto: %77
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v473;
        // pto: %77
        ;
        uint64_t v474 = (uint64_t) v37;
        TASSIGN(v473, v474);
        // pto: %t__tmp_v269
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v475 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %t__tmp_v269
        ;
        uint64_t v476 = (uint64_t) v43;
        TASSIGN(v475, v476);
        pipe_barrier(PIPE_V);
        TADD(v475, v467, v473);
        // pto: %col4_inline1213__ssa_v3
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v477 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %col4_inline1213__ssa_v3
        ;
        uint64_t v478 = (uint64_t) v43;
        TASSIGN(v477, v478);
        pipe_barrier(PIPE_V);
        TADDS(v477, v475, v80);
        // pto: %cur_inline1151__ssa_v3
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v479 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
        // pto: %cur_inline1151__ssa_v3
        ;
        uint64_t v480 = (uint64_t) v34;
        TASSIGN(v479, v480);
        pipe_barrier(PIPE_V);
        TDIV(v479, v459, v477);
      };
      // pto: %row_major_inline1089__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v481 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %row_major_inline1089__ssa_v0
      ;
      uint64_t v482 = (uint64_t) v45;
      TASSIGN(v481, v482);
      pipe_barrier(PIPE_V);
      TGATHER(v481, v442, v340, v320);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
      // pto: %t__tmp_v270
      ;
      Tile<TileType::Vec, float, 4, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v483 = Tile<TileType::Vec, float, 4, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v77, v70);
      // pto: %t__tmp_v270
      ;
      uint64_t v484 = (uint64_t) v45;
      TASSIGN(v483, v484);
      v483.SetValidShape(v225, v70);
      // pto: %comb_inline1100__ssa_v0_pview
      ;
      const int64_t v485 = 0;
      // pto: %comb_inline1100__ssa_v0_pview
      ;
      __gm__ float* v486 = PTOAS__GLOBAL_TENSOR_DATA(v148);
      // pto: %comb_inline1100__ssa_v0_pview
      ;
      const int64_t v487 = 1;
      // pto: %comb_inline1100__ssa_v0_pview
      ;
      const int64_t v488 = 1;
      // pto: %comb_inline1100__ssa_v0_pview
      ;
      const int64_t v489 = 1;
      // pto: %comb_inline1100__ssa_v0_pview
      ;
      int64_t v490 = v225 * v70;
      // pto: %comb_inline1100__ssa_v0_pview
      ;
      int64_t v491 = v489 * v490;
      // pto: %comb_inline1100__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, 16> v492 = pto::Shape<1, 1, 1, -1, 16>(v487, v488, v489, v225, v70);
      // pto: %comb_inline1100__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v493 = pto::Stride<-1, -1, -1, -1, -1>(v488 * v491, v491, v490, v70, v65);
      // pto: %comb_inline1100__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v494 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v486 + (v485 + v255 * v70 + v74 * v65), v492, v493);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
      TSTORE(v494, v483);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    };
  } else {
    // pto: %79
    ;
    int64_t v495 = (int64_t) ((uint64_t) v215 - (uint64_t) v17);
    // pto: %81
    ;
    int64_t v496 = v495 < v21 ? v495 : v21;
    // pto: %80, %82, %83
    ;
    int64_t v497 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v495 * (uint64_t) v20) + (uint64_t) v496) * (uint64_t) v75);
    // pto: %84
    ;
    int64_t v498 = (int64_t) ((uint64_t) v21 - (uint64_t) v495);
    // pto: %85
    ;
    int64_t v499 = v498 < v74 ? v74 : v498;
    // pto: %86
    ;
    int64_t v500 = v499 < v65 ? v499 : v65;
    // pto: %89, %87, %88
    ;
    int64_t v501 = (int64_t) ((uint64_t) v497 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v20 + (uint64_t) v500) * (uint64_t) v75));
    // pto: %90
    ;
    // pto: %91
    ;
    // pto: %d_begin_inline1073__phi_v2
    ;
    int64_t v502;
    // pto: %work_begin_inline1269__phi_v3
    ;
    int64_t v503;
    // pto: %work_end_inline999__phi_v3
    ;
    int64_t v504;
    if (v23 < v24) {
      // pto: %92
      ;
      // pto: %d_worker_inline1018__phi_v2
      ;
      int64_t v505;
      // pto: %d_workers_inline1147__phi_v2
      ;
      int64_t v506;
      // pto: %pair_id_inline1271__phi_v2
      ;
      int64_t v507;
      if (v495 < v25) {
        // pto: %93
        ;
        int64_t v508 = (int64_t) ((uint64_t) v26 + (uint64_t) v65);
        // pto: %94
        ;
        // pto: %96
        ;
        v505 = v495 % v508;
        v506 = v508;
        v507 = v495 / v508;
      } else {
        // pto: %98
        ;
        int64_t v509 = (int64_t) ((uint64_t) v495 - (uint64_t) v25);
        // pto: %100, %99
        ;
        // pto: %101
        ;
        v505 = v509 % v26;
        v506 = v26;
        v507 = (int64_t) ((uint64_t) v27 + (uint64_t) (v509 / v26));
      };
      // pto: %102
      ;
      uint64_t v510 = (uint64_t) v507;
      // pto: %102
      ;
      int64_t v511 = (int64_t) (v510 * (uint64_t) v75);
      // pto: %103
      ;
      int64_t v512 = v22 < v511 ? v22 : v511;
      // pto: %104
      ;
      int64_t v513 = (int64_t) ((uint64_t) v512 + (uint64_t) v75);
      // pto: %105
      ;
      // pto: %106
      ;
      int64_t v514 = v69 / v506;
      // pto: %107
      ;
      int64_t v515 = v69 % v506;
      // pto: %108
      ;
      uint64_t v516 = (uint64_t) v505;
      // pto: %109
      ;
      bool v517 = v505 < v515;
      // pto: %109
      ;
      int64_t v518 = v517 ? v505 : v515;
      // pto: %108, %110
      ;
      v502 = (int64_t) ((uint64_t) ((int64_t) v516 * (uint64_t) v514) + (uint64_t) v518);
      v503 = v512;
      v504 = v22 < v513 ? v22 : v513;
    } else {
      v502 = v74;
      v503 = v497;
      v504 = v22 < v501 ? v22 : v501;
    };
    // pto: %111
    ;
    bool v519 = v24 <= v495;
    // pto: %work_begin_inline1269__phi_v5
    ;
    int64_t v520 = v519 ? v74 : v503;
    // pto: %work_end_inline999__phi_v5
    ;
    int64_t v521 = v519 ? v74 : v504;
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    for (int64_t v522 = v520; v522 < v521; v522 += v77) {
      // pto: %112
      ;
      int64_t v523 = (int64_t) ((uint64_t) v521 - (uint64_t) v522);
      // pto: %113
      ;
      int64_t v524 = v523 < v77 ? v523 : v77;
      // pto: %114
      ;
      int64_t v525 = (int64_t) ((uint64_t) v524 * (uint64_t) v64);
      // pto: %m_parts_load_inline1277__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v526 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v525);
      // pto: %m_parts_load_inline1277__ssa_v0
      ;
      uint64_t v527 = (uint64_t) v34;
      TASSIGN(v526, v527);
      // pto: %115
      ;
      int64_t v528 = (int64_t) ((uint64_t) v522 * (uint64_t) v64);
      // pto: %116
      ;
      int64_t v529 = v528 < v74 ? v74 : v528;
      // pto: %118
      ;
      const int64_t v530 = 0;
      // pto: %118
      ;
      __gm__ float* v531 = PTOAS__GLOBAL_TENSOR_DATA(v122);
      // pto: %118
      ;
      const int64_t v532 = 1;
      // pto: %118
      ;
      const int64_t v533 = 1;
      // pto: %118
      ;
      const int64_t v534 = 1;
      // pto: %118
      ;
      int64_t v535 = v69 * v114;
      // pto: %118
      ;
      int64_t v536 = v534 * v535;
      // pto: %118
      ;
      pto::Shape<1, 1, 1, 8, -1> v537 = pto::Shape<1, 1, 1, 8, -1>(v532, v533, v534, v69, v525);
      // pto: %118
      ;
      pto::Stride<-1, -1, -1, -1, -1> v538 = pto::Stride<-1, -1, -1, -1, -1>(v533 * v536, v536, v535, v114, v65);
      // pto: %118
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v539 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v531 + (v530 + v74 * v114 + v529 * v65), v537, v538);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      TLOAD(v526, v539);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      // pto: %t__tmp_v271
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v540 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v69, v78);
      // pto: %t__tmp_v271
      ;
      uint64_t v541 = (uint64_t) v34;
      TASSIGN(v540, v541);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      TFILLPAD<pto::TFillPadMode::InPlace>(v540, v526);
      v540.SetValidShape(v69, v78);
      // pto: %m_parts_tmp_inline1059__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v542 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v78);
      // pto: %m_parts_tmp_inline1059__ssa_v0
      ;
      uint64_t v543 = (uint64_t) v36;
      TASSIGN(v542, v543);
      // pto: %t__tmp_v272
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v544 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v78);
      // pto: %t__tmp_v272
      ;
      uint64_t v545 = (uint64_t) v34;
      TASSIGN(v544, v545);
      // pto: %119
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, 8, 256, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v546;
      TRESHAPE(v546, v540);
      // pto: %120
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, 8, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v547;
      TRESHAPE(v547, v542);
      // pto: %121
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v548;
      TRESHAPE(v548, v544);
      pipe_barrier(PIPE_V);
      const bool v549 = true;
      TCOLSUM(v548, v546, v547, v549);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      // pto: %m_total_inline1249__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v550 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v64);
      // pto: %m_total_inline1249__ssa_v0
      ;
      uint64_t v551 = (uint64_t) v34;
      TASSIGN(v550, v551);
      // pto: %m_sq_load_inline1279__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v552 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v524);
      // pto: %m_sq_load_inline1279__ssa_v0
      ;
      uint64_t v553 = (uint64_t) v37;
      TASSIGN(v552, v553);
      // pto: %122
      ;
      int64_t v554 = v522 < v74 ? v74 : v522;
      // pto: %123
      ;
      int64_t v555 = (int64_t) v28;
      // pto: %123
      ;
      const int64_t v556 = 0;
      // pto: %123
      ;
      __gm__ float* v557 = PTOAS__GLOBAL_TENSOR_DATA(v131);
      // pto: %123
      ;
      const int64_t v558 = 1;
      // pto: %123
      ;
      const int64_t v559 = 1;
      // pto: %123
      ;
      const int64_t v560 = 1;
      // pto: %123
      ;
      int64_t v561 = v70 * v555;
      // pto: %123
      ;
      int64_t v562 = v560 * v561;
      // pto: %123
      ;
      pto::Shape<1, 1, 1, 16, -1> v563 = pto::Shape<1, 1, 1, 16, -1>(v558, v559, v560, v70, v524);
      // pto: %123
      ;
      pto::Stride<-1, -1, -1, -1, -1> v564 = pto::Stride<-1, -1, -1, -1, -1>(v559 * v562, v562, v561, v555, v65);
      // pto: %123
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v565 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v557 + (v556 + v74 * v555 + v554 * v65), v563, v564);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      TLOAD(v552, v565);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      // pto: %t__tmp_v273
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v566 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v70, v69);
      // pto: %t__tmp_v273
      ;
      uint64_t v567 = (uint64_t) v37;
      TASSIGN(v566, v567);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      TFILLPAD<pto::TFillPadMode::InPlace>(v566, v552);
      v566.SetValidShape(v70, v69);
      // pto: %m_sq_tmp_inline1234__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v568 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70, v69);
      // pto: %m_sq_tmp_inline1234__ssa_v0
      ;
      uint64_t v569 = (uint64_t) v38;
      TASSIGN(v568, v569);
      // pto: %m_rms_tmp_inline1244__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v570 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %m_rms_tmp_inline1244__ssa_v0
      ;
      uint64_t v571 = (uint64_t) v39;
      TASSIGN(v570, v571);
      // pto: %t__tmp_v274
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v572 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v274
      ;
      uint64_t v573 = (uint64_t) v37;
      TASSIGN(v572, v573);
      // pto: %124
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, 16, 8, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v574;
      TRESHAPE(v574, v566);
      // pto: %125
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, 16, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v575;
      TRESHAPE(v575, v568);
      // pto: %126
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v576;
      TRESHAPE(v576, v572);
      pipe_barrier(PIPE_V);
      const bool v577 = true;
      TCOLSUM(v576, v574, v575, v577);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      // pto: %t__tmp_v275
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v578 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v275
      ;
      uint64_t v579 = (uint64_t) v37;
      TASSIGN(v578, v579);
      pipe_barrier(PIPE_V);
      TMULS(v578, v572, v79);
      // pto: %t__tmp_v276
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v580 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v276
      ;
      uint64_t v581 = (uint64_t) v37;
      TASSIGN(v580, v581);
      pipe_barrier(PIPE_V);
      TADDS(v580, v578, v80);
      // pto: %t__tmp_v277
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v582 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v277
      ;
      uint64_t v583 = (uint64_t) v46;
      TASSIGN(v582, v583);
      pipe_barrier(PIPE_V);
      TRSQRT(v582, v580, v570);
      // pto: %m_inv_inline1280__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v584 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v65);
      // pto: %m_inv_inline1280__ssa_v0
      ;
      uint64_t v585 = (uint64_t) v46;
      TASSIGN(v584, v585);
      // pto: %gate_base_inline1281__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v586 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %gate_base_inline1281__ssa_v0
      ;
      uint64_t v587 = (uint64_t) v50;
      TASSIGN(v586, v587);
      // pto: %127
      ;
      __gm__ float* v588 = PTOAS__GLOBAL_TENSOR_DATA(v139);
      // pto: %127
      ;
      pto::Shape<1, 1, 1, 1, 8> v589 = pto::Shape<1, 1, 1, 1, 8>();
      // pto: %127
      ;
      pto::Stride<24, 24, 24, 24, 1> v590 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %127
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v591 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v588, v589, v590);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      TLOAD(v586, v591);
      // pto: %gate_scales_inline1283__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v592 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v68);
      // pto: %gate_scales_inline1283__ssa_v0
      ;
      uint64_t v593 = (uint64_t) v51;
      TASSIGN(v592, v593);
      // pto: %hc_scale_2d_inline1168__ssa_v0_pview
      ;
      __gm__ float* v594 = PTOAS__GLOBAL_TENSOR_DATA(v156);
      // pto: %hc_scale_2d_inline1168__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 3> v595 = pto::Shape<1, 1, 1, 1, 3>();
      // pto: %hc_scale_2d_inline1168__ssa_v0_pview
      ;
      pto::Stride<3, 3, 3, 3, 1> v596 = pto::Stride<3, 3, 3, 3, 1>();
      // pto: %hc_scale_2d_inline1168__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND> v597 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND>(v594, v595, v596);
      TLOAD(v592, v597);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      // pto: %t__ci_tmp_v2
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v598 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v83);
      // pto: %t__ci_tmp_v2
      ;
      uint64_t v599 = (uint64_t) v37;
      TASSIGN(v598, v599);
      // pto: %t__tmp_v278
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v600 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v278
      ;
      uint64_t v601 = (uint64_t) v52;
      TASSIGN(v600, v601);
      // pto: %128
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v602;
      TRESHAPE(v602, v598);
      // pto: %129
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v603;
      TRESHAPE(v603, v600);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v603, v72, v602);
      // pto: %gate_ids_inline1284__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v604 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %gate_ids_inline1284__ssa_v0
      ;
      uint64_t v605 = (uint64_t) v37;
      TASSIGN(v604, v605);
      pipe_barrier(PIPE_V);
      RoundMode v606 = RoundMode::CAST_ROUND;
      SaturationMode v607 = SaturationMode::ON;
      TCVT(v604, v600, v606, v607);
      // pto: %t__tmp_v279
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v608 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %t__tmp_v279
      ;
      uint64_t v609 = (uint64_t) v37;
      TASSIGN(v608, v609);
      pipe_barrier(PIPE_V);
      TMULS(v608, v604, v92);
      // pto: %gate_scale_ids_inline1287__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v610 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %gate_scale_ids_inline1287__ssa_v0
      ;
      uint64_t v611 = (uint64_t) v37;
      TASSIGN(v610, v611);
      pipe_barrier(PIPE_V);
      RoundMode v612 = RoundMode::CAST_FLOOR;
      SaturationMode v613 = SaturationMode::ON;
      TCVT(v610, v608, v612, v613);
      // pto: %gate_tmp_inline1133__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v614 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %gate_tmp_inline1133__ssa_v0
      ;
      uint64_t v615 = (uint64_t) v53;
      TASSIGN(v614, v615);
      // pto: %gate_scale_vec_inline1260__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v616 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v69);
      // pto: %gate_scale_vec_inline1260__ssa_v0
      ;
      uint64_t v617 = (uint64_t) v40;
      TASSIGN(v616, v617);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      TGATHER(v616, v592, v610, v614);
      // pto: %t__tmp_v280
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v618 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v69);
      // pto: %t__tmp_v280
      ;
      uint64_t v619 = (uint64_t) v34;
      TASSIGN(v618, v619);
      // pto: %130
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v620;
      // pto: %130
      ;
      uint64_t v621 = (uint64_t) v34;
      TASSIGN(v620, v621);
      // pto: %gate_raw_inline1026__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v622 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v69);
      // pto: %gate_raw_inline1026__ssa_v0
      ;
      uint64_t v623 = (uint64_t) v34;
      TASSIGN(v622, v623);
      TADDS(v622, v620, v81);
      // pto: %gate_norm_inline1216__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v624 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v69);
      // pto: %gate_norm_inline1216__ssa_v0
      ;
      uint64_t v625 = (uint64_t) v34;
      TASSIGN(v624, v625);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v624, v622, v584);
      // pto: %gate_scaled_inline1253__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v626 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v69);
      // pto: %gate_scaled_inline1253__ssa_v0
      ;
      uint64_t v627 = (uint64_t) v34;
      TASSIGN(v626, v627);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v626, v624, v616);
      // pto: %t__tmp_v281
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v628 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v69);
      // pto: %t__tmp_v281
      ;
      uint64_t v629 = (uint64_t) v45;
      TASSIGN(v628, v629);
      TCOLEXPAND(v628, v586);
      // pto: %gate_logits_inline1042__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v630 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v69);
      // pto: %gate_logits_inline1042__ssa_v0
      ;
      uint64_t v631 = (uint64_t) v34;
      TASSIGN(v630, v631);
      pipe_barrier(PIPE_V);
      TADD(v630, v626, v628);
      // pto: %t__tmp_v282
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v632 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v69);
      // pto: %t__tmp_v282
      ;
      uint64_t v633 = (uint64_t) v34;
      TASSIGN(v632, v633);
      pipe_barrier(PIPE_V);
      TNEG(v632, v630);
      // pto: %t__tmp_v283
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v634 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v69);
      // pto: %t__tmp_v283
      ;
      uint64_t v635 = (uint64_t) v34;
      TASSIGN(v634, v635);
      pipe_barrier(PIPE_V);
      TEXP(v634, v632);
      // pto: %t__tmp_v284
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v636 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v69);
      // pto: %t__tmp_v284
      ;
      uint64_t v637 = (uint64_t) v34;
      TASSIGN(v636, v637);
      pipe_barrier(PIPE_V);
      TADDS(v636, v634, v93);
      // pto: %gate_sigmoid_inline994__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v638 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v69, v69);
      // pto: %gate_sigmoid_inline994__ssa_v0
      ;
      uint64_t v639 = (uint64_t) v54;
      TASSIGN(v638, v639);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
      TRECIP(v638, v636);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      // pto: %t__ci_tmp_v3
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v640 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v83);
      // pto: %t__ci_tmp_v3
      ;
      uint64_t v641 = (uint64_t) v34;
      TASSIGN(v640, v641);
      // pto: %t__tmp_v285
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v642 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v285
      ;
      uint64_t v643 = (uint64_t) v43;
      TASSIGN(v642, v643);
      // pto: %131
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v644;
      TRESHAPE(v644, v640);
      // pto: %132
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v645;
      TRESHAPE(v645, v642);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v645, v72, v644);
      // pto: %post_ids_float_inline1273__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v646 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %post_ids_float_inline1273__ssa_v0
      ;
      uint64_t v647 = (uint64_t) v34;
      TASSIGN(v646, v647);
      pipe_barrier(PIPE_V);
      RoundMode v648 = RoundMode::CAST_ROUND;
      SaturationMode v649 = SaturationMode::ON;
      TCVT(v646, v642, v648, v649);
      // pto: %t__tmp_v286
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v650 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v286
      ;
      uint64_t v651 = (uint64_t) v55;
      TASSIGN(v650, v651);
      pipe_barrier(PIPE_V);
      TMULS(v650, v646, v92);
      // pto: %t__tmp_v287
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v652 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v287
      ;
      uint64_t v653 = (uint64_t) v55;
      TASSIGN(v652, v653);
      pipe_barrier(PIPE_V);
      RoundMode v654 = RoundMode::CAST_FLOOR;
      SaturationMode v655 = SaturationMode::ON;
      TCVT(v652, v650, v654, v655);
      // pto: %post_rows_inline1082__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v656 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %post_rows_inline1082__ssa_v0
      ;
      uint64_t v657 = (uint64_t) v55;
      TASSIGN(v656, v657);
      pipe_barrier(PIPE_V);
      RoundMode v658 = RoundMode::CAST_ROUND;
      SaturationMode v659 = SaturationMode::ON;
      TCVT(v656, v652, v658, v659);
      // pto: %t__tmp_v288
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v660 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v288
      ;
      uint64_t v661 = (uint64_t) v55;
      TASSIGN(v660, v661);
      pipe_barrier(PIPE_V);
      TMULS(v660, v656, v94);
      // pto: %t__tmp_v289
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v662 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v289
      ;
      uint64_t v663 = (uint64_t) v34;
      TASSIGN(v662, v663);
      pipe_barrier(PIPE_V);
      TADD(v662, v646, v660);
      // pto: %t__tmp_v290
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v664 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v290
      ;
      uint64_t v665 = (uint64_t) v34;
      TASSIGN(v664, v665);
      pipe_barrier(PIPE_V);
      TADDS(v664, v662, v94);
      // pto: %post_ids_inline993__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v666 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %post_ids_inline993__ssa_v0
      ;
      uint64_t v667 = (uint64_t) v34;
      TASSIGN(v666, v667);
      pipe_barrier(PIPE_V);
      RoundMode v668 = RoundMode::CAST_ROUND;
      SaturationMode v669 = SaturationMode::ON;
      TCVT(v666, v664, v668, v669);
      // pto: %post_tmp_inline992__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v670 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %post_tmp_inline992__ssa_v0
      ;
      uint64_t v671 = (uint64_t) v55;
      TASSIGN(v670, v671);
      // pto: %t__tmp_v291
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v672 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v82);
      // pto: %t__tmp_v291
      ;
      uint64_t v673 = (uint64_t) v54;
      TASSIGN(v672, v673);
      // pto: %post_values_inline1006__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v674 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %post_values_inline1006__ssa_v0
      ;
      uint64_t v675 = (uint64_t) v45;
      TASSIGN(v674, v675);
      pipe_barrier(PIPE_V);
      TGATHER(v674, v672, v666, v670);
      // pto: %t__tmp_v292
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v676 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v64);
      // pto: %t__tmp_v292
      ;
      uint64_t v677 = (uint64_t) v34;
      TASSIGN(v676, v677);
      pipe_barrier(PIPE_V);
      TMULS(v676, v674, v95);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      // pto: %133
      ;
      int64_t v678 = (int64_t) ((uint64_t) v524 * (uint64_t) v77);
      v676.SetValidShape(v65, v678);
      // pto: %134
      ;
      bool v679 = v502 == v74;
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      if (v679) {
        // pto: %136
        ;
        int64_t v680 = (int64_t) ((uint64_t) v522 * (uint64_t) v77);
        // pto: %137
        ;
        int64_t v681 = v680 < v74 ? v74 : v680;
        // pto: %post_flat_inline1112__ssa_v0_pview
        ;
        const int64_t v682 = 0;
        // pto: %post_flat_inline1112__ssa_v0_pview
        ;
        __gm__ float* v683 = PTOAS__GLOBAL_TENSOR_DATA(v165);
        // pto: %post_flat_inline1112__ssa_v0_pview
        ;
        const int64_t v684 = 1;
        // pto: %post_flat_inline1112__ssa_v0_pview
        ;
        const int64_t v685 = 1;
        // pto: %post_flat_inline1112__ssa_v0_pview
        ;
        const int64_t v686 = 1;
        // pto: %post_flat_inline1112__ssa_v0_pview
        ;
        int64_t v687 = v65 * v157;
        // pto: %post_flat_inline1112__ssa_v0_pview
        ;
        int64_t v688 = v686 * v687;
        // pto: %post_flat_inline1112__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, -1> v689 = pto::Shape<1, 1, 1, 1, -1>(v684, v685, v686, v65, v678);
        // pto: %post_flat_inline1112__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v690 = pto::Stride<-1, -1, -1, -1, -1>(v685 * v688, v688, v687, v157, v65);
        // pto: %post_flat_inline1112__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v691 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v683 + (v682 + v74 * v157 + v681 * v65), v689, v690);
        TSTORE(v691, v676);
      };
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      // pto: %138
      ;
      int64_t v692 = v524 / v75;
      // pto: %139
      ;
      int64_t v693 = (int64_t) ((uint64_t) v692 * (uint64_t) v75);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      for (int64_t v694 = v74; v694 < v693; v694 += v75) {
        // pto: %140
        ;
        int64_t v695 = (int64_t) ((uint64_t) v522 + (uint64_t) v694);
        // pto: %141
        ;
        int64_t v696 = (int64_t) ((uint64_t) v694 + (uint64_t) v65);
        // pto: %142
        ;
        int64_t v697 = (int64_t) ((uint64_t) v522 + (uint64_t) v696);
        // pto: %flat_offset_mul
        ;
        int64_t v698 = (int64_t) ((uint64_t) v694 * (uint64_t) v69);
        // pto: %t__tmp_v293
        ;
        float v699 = v638.GetValue(v698);
        // pto: %143
        ;
        float v700 = v699 + v80;
        // pto: %145, %t__tmp_v294
        ;
        float v701 = v638.GetValue((int64_t) ((uint64_t) v698 + (uint64_t) v65));
        // pto: %146
        ;
        float v702 = v701 + v80;
        // pto: %148, %t__tmp_v295
        ;
        float v703 = v638.GetValue((int64_t) ((uint64_t) v698 + (uint64_t) v75));
        // pto: %149
        ;
        float v704 = v703 + v80;
        // pto: %151, %t__tmp_v296
        ;
        float v705 = v638.GetValue((int64_t) ((uint64_t) v698 + (uint64_t) v68));
        // pto: %152
        ;
        float v706 = v705 + v80;
        // pto: %t__tmp_v297
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v707 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %t__tmp_v297
        ;
        uint64_t v708 = (uint64_t) v56;
        TASSIGN(v707, v708);
        // pto: %153
        ;
        int64_t v709 = v695 < v74 ? v74 : v695;
        // pto: %154
        ;
        __gm__ bfloat16_t* v710 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %154
        ;
        const int64_t v711 = 0;
        // pto: %154
        ;
        const int64_t v712 = 16384;
        // pto: %154
        ;
        pto::Shape<1, 1, 1, 1, 4096> v713 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %154
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v714 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %154
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v715 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v710 + (v711 + v709 * v712), v713, v714);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v707, v715);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        // pto: %t__tmp_v298
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v716 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %t__tmp_v298
        ;
        uint64_t v717 = (uint64_t) v57;
        TASSIGN(v716, v717);
        // pto: %156
        ;
        __gm__ bfloat16_t* v718 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %156
        ;
        const int64_t v719 = 4096;
        // pto: %156
        ;
        const int64_t v720 = 16384;
        // pto: %156
        ;
        pto::Shape<1, 1, 1, 1, 4096> v721 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %156
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v722 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %156
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v723 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v718 + (v719 + v709 * v720), v721, v722);
        TLOAD(v716, v723);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %t__tmp_v299
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v724 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %t__tmp_v299
        ;
        uint64_t v725 = (uint64_t) v58;
        TASSIGN(v724, v725);
        // pto: %158
        ;
        __gm__ bfloat16_t* v726 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %158
        ;
        const int64_t v727 = 8192;
        // pto: %158
        ;
        const int64_t v728 = 16384;
        // pto: %158
        ;
        pto::Shape<1, 1, 1, 1, 4096> v729 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %158
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v730 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %158
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v731 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v726 + (v727 + v709 * v728), v729, v730);
        TLOAD(v724, v731);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %t__tmp_v300
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v732 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %t__tmp_v300
        ;
        uint64_t v733 = (uint64_t) v59;
        TASSIGN(v732, v733);
        // pto: %160
        ;
        __gm__ bfloat16_t* v734 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %160
        ;
        const int64_t v735 = 12288;
        // pto: %160
        ;
        const int64_t v736 = 16384;
        // pto: %160
        ;
        pto::Shape<1, 1, 1, 1, 4096> v737 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %160
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v738 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %160
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v739 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v734 + (v735 + v709 * v736), v737, v738);
        TLOAD(v732, v739);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %163
        ;
        int64_t v740 = (int64_t) ((uint64_t) v696 * (uint64_t) v69);
        // pto: %161
        ;
        float v741 = v638.GetValue(v740);
        // pto: %165
        ;
        float v742 = v741 + v80;
        // pto: %169, %166
        ;
        float v743 = v638.GetValue((int64_t) ((uint64_t) v740 + (uint64_t) v65));
        // pto: %170
        ;
        float v744 = v743 + v80;
        // pto: %174, %171
        ;
        float v745 = v638.GetValue((int64_t) ((uint64_t) v740 + (uint64_t) v75));
        // pto: %175
        ;
        float v746 = v745 + v80;
        // pto: %179, %176
        ;
        float v747 = v638.GetValue((int64_t) ((uint64_t) v740 + (uint64_t) v68));
        // pto: %180
        ;
        float v748 = v747 + v80;
        // pto: %0
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v749 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %0
        ;
        uint64_t v750 = (uint64_t) v33;
        TASSIGN(v749, v750);
        // pto: %181
        ;
        int64_t v751 = v697 < v74 ? v74 : v697;
        // pto: %182
        ;
        __gm__ bfloat16_t* v752 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %182
        ;
        const int64_t v753 = 0;
        // pto: %182
        ;
        const int64_t v754 = 16384;
        // pto: %182
        ;
        pto::Shape<1, 1, 1, 1, 4096> v755 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %182
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v756 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %182
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v757 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v752 + (v753 + v751 * v754), v755, v756);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        TLOAD(v749, v757);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %1
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v758 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %1
        ;
        uint64_t v759 = (uint64_t) v60;
        TASSIGN(v758, v759);
        // pto: %184
        ;
        __gm__ bfloat16_t* v760 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %184
        ;
        const int64_t v761 = 4096;
        // pto: %184
        ;
        const int64_t v762 = 16384;
        // pto: %184
        ;
        pto::Shape<1, 1, 1, 1, 4096> v763 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %184
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v764 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %184
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v765 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v760 + (v761 + v751 * v762), v763, v764);
        TLOAD(v758, v765);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        // pto: %2
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v766 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %2
        ;
        uint64_t v767 = (uint64_t) v61;
        TASSIGN(v766, v767);
        // pto: %186
        ;
        __gm__ bfloat16_t* v768 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %186
        ;
        const int64_t v769 = 8192;
        // pto: %186
        ;
        const int64_t v770 = 16384;
        // pto: %186
        ;
        pto::Shape<1, 1, 1, 1, 4096> v771 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %186
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v772 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %186
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v773 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v768 + (v769 + v751 * v770), v771, v772);
        TLOAD(v766, v773);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        // pto: %3
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v774 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %3
        ;
        uint64_t v775 = (uint64_t) v62;
        TASSIGN(v774, v775);
        // pto: %188
        ;
        __gm__ bfloat16_t* v776 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %188
        ;
        const int64_t v777 = 12288;
        // pto: %188
        ;
        const int64_t v778 = 16384;
        // pto: %188
        ;
        pto::Shape<1, 1, 1, 1, 4096> v779 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %188
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v780 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %188
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v781 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v776 + (v777 + v751 * v778), v779, v780);
        TLOAD(v774, v781);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        // pto: %scalar_x0_inline1222__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v782 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %scalar_x0_inline1222__ssa_v0
        ;
        uint64_t v783 = (uint64_t) v34;
        TASSIGN(v782, v783);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        RoundMode v784 = RoundMode::CAST_ROUND;
        SaturationMode v785 = SaturationMode::ON;
        TCVT(v782, v707, v784, v785);
        // pto: %scalar_y0_inline980__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v786 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %scalar_y0_inline980__ssa_v0
        ;
        uint64_t v787 = (uint64_t) v34;
        TASSIGN(v786, v787);
        pipe_barrier(PIPE_V);
        TMULS(v786, v782, v700);
        // pto: %scalar_x1_inline979__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v788 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %scalar_x1_inline979__ssa_v0
        ;
        uint64_t v789 = (uint64_t) v56;
        TASSIGN(v788, v789);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v790 = RoundMode::CAST_ROUND;
        SaturationMode v791 = SaturationMode::ON;
        TCVT(v788, v716, v790, v791);
        // pto: %scalar_y1_inline1258__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v792 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %scalar_y1_inline1258__ssa_v0
        ;
        uint64_t v793 = (uint64_t) v56;
        TASSIGN(v792, v793);
        pipe_barrier(PIPE_V);
        TMULS(v792, v788, v702);
        // pto: %scalar_sum01_inline1071__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v794 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %scalar_sum01_inline1071__ssa_v0
        ;
        uint64_t v795 = (uint64_t) v34;
        TASSIGN(v794, v795);
        pipe_barrier(PIPE_V);
        TADD(v794, v786, v792);
        // pto: %scalar_x2_inline1262__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v796 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %scalar_x2_inline1262__ssa_v0
        ;
        uint64_t v797 = (uint64_t) v56;
        TASSIGN(v796, v797);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        RoundMode v798 = RoundMode::CAST_ROUND;
        SaturationMode v799 = SaturationMode::ON;
        TCVT(v796, v724, v798, v799);
        // pto: %scalar_y2_inline1053__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v800 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %scalar_y2_inline1053__ssa_v0
        ;
        uint64_t v801 = (uint64_t) v56;
        TASSIGN(v800, v801);
        pipe_barrier(PIPE_V);
        TMULS(v800, v796, v704);
        // pto: %scalar_x3_inline978__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v802 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %scalar_x3_inline978__ssa_v0
        ;
        uint64_t v803 = (uint64_t) v57;
        TASSIGN(v802, v803);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        RoundMode v804 = RoundMode::CAST_ROUND;
        SaturationMode v805 = SaturationMode::ON;
        TCVT(v802, v732, v804, v805);
        // pto: %scalar_y3_inline977__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v806 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %scalar_y3_inline977__ssa_v0
        ;
        uint64_t v807 = (uint64_t) v57;
        TASSIGN(v806, v807);
        pipe_barrier(PIPE_V);
        TMULS(v806, v802, v706);
        // pto: %t__tmp_v301
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v808 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %t__tmp_v301
        ;
        uint64_t v809 = (uint64_t) v56;
        TASSIGN(v808, v809);
        pipe_barrier(PIPE_V);
        TADD(v808, v800, v806);
        // pto: %scalar_result_inline976__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v810 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %scalar_result_inline976__ssa_v0
        ;
        uint64_t v811 = (uint64_t) v34;
        TASSIGN(v810, v811);
        pipe_barrier(PIPE_V);
        TADD(v810, v794, v808);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %scalar_bf16_inline975__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v812 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %scalar_bf16_inline975__ssa_v0
        ;
        uint64_t v813 = (uint64_t) v34;
        TASSIGN(v812, v813);
        pipe_barrier(PIPE_V);
        RoundMode v814 = RoundMode::CAST_RINT;
        SaturationMode v815 = SaturationMode::ON;
        TCVT(v812, v810, v814, v815);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
        // pto: %mixed_view_inline1121__ssa_v0_pview
        ;
        __gm__ bfloat16_t* v816 = PTOAS__GLOBAL_TENSOR_DATA(v174);
        // pto: %mixed_view_inline1121__ssa_v0_pview
        ;
        const int64_t v817 = 0;
        // pto: %mixed_view_inline1121__ssa_v0_pview
        ;
        const int64_t v818 = 4096;
        // pto: %mixed_view_inline1121__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, 4096> v819 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %mixed_view_inline1121__ssa_v0_pview
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v820 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %mixed_view_inline1121__ssa_v0_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v821 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v816 + (v817 + v709 * v818), v819, v820);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v821, v812);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        // pto: %4
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v822 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %4
        ;
        uint64_t v823 = (uint64_t) v63;
        TASSIGN(v822, v823);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        RoundMode v824 = RoundMode::CAST_ROUND;
        SaturationMode v825 = SaturationMode::ON;
        TCVT(v822, v749, v824, v825);
        // pto: %5
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v826 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %5
        ;
        uint64_t v827 = (uint64_t) v63;
        TASSIGN(v826, v827);
        pipe_barrier(PIPE_V);
        TMULS(v826, v822, v742);
        // pto: %6
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v828 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %6
        ;
        uint64_t v829 = (uint64_t) v33;
        TASSIGN(v828, v829);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        RoundMode v830 = RoundMode::CAST_ROUND;
        SaturationMode v831 = SaturationMode::ON;
        TCVT(v828, v758, v830, v831);
        // pto: %7
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v832 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %7
        ;
        uint64_t v833 = (uint64_t) v33;
        TASSIGN(v832, v833);
        pipe_barrier(PIPE_V);
        TMULS(v832, v828, v744);
        // pto: %8
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v834 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %8
        ;
        uint64_t v835 = (uint64_t) v63;
        TASSIGN(v834, v835);
        pipe_barrier(PIPE_V);
        TADD(v834, v826, v832);
        // pto: %9
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v836 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %9
        ;
        uint64_t v837 = (uint64_t) v33;
        TASSIGN(v836, v837);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        RoundMode v838 = RoundMode::CAST_ROUND;
        SaturationMode v839 = SaturationMode::ON;
        TCVT(v836, v766, v838, v839);
        // pto: %10
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v840 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %10
        ;
        uint64_t v841 = (uint64_t) v33;
        TASSIGN(v840, v841);
        pipe_barrier(PIPE_V);
        TMULS(v840, v836, v746);
        // pto: %11
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v842 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %11
        ;
        uint64_t v843 = (uint64_t) v60;
        TASSIGN(v842, v843);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        RoundMode v844 = RoundMode::CAST_ROUND;
        SaturationMode v845 = SaturationMode::ON;
        TCVT(v842, v774, v844, v845);
        // pto: %12
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v846 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %12
        ;
        uint64_t v847 = (uint64_t) v60;
        TASSIGN(v846, v847);
        pipe_barrier(PIPE_V);
        TMULS(v846, v842, v748);
        // pto: %13
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v848 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %13
        ;
        uint64_t v849 = (uint64_t) v33;
        TASSIGN(v848, v849);
        pipe_barrier(PIPE_V);
        TADD(v848, v840, v846);
        // pto: %14
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v850 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %14
        ;
        uint64_t v851 = (uint64_t) v63;
        TASSIGN(v850, v851);
        pipe_barrier(PIPE_V);
        TADD(v850, v834, v848);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        // pto: %15
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v852 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %15
        ;
        uint64_t v853 = (uint64_t) v63;
        TASSIGN(v852, v853);
        pipe_barrier(PIPE_V);
        RoundMode v854 = RoundMode::CAST_RINT;
        SaturationMode v855 = SaturationMode::ON;
        TCVT(v852, v850, v854, v855);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
        // pto: %192
        ;
        __gm__ bfloat16_t* v856 = PTOAS__GLOBAL_TENSOR_DATA(v174);
        // pto: %192
        ;
        const int64_t v857 = 0;
        // pto: %192
        ;
        const int64_t v858 = 4096;
        // pto: %192
        ;
        pto::Shape<1, 1, 1, 1, 4096> v859 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %192
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v860 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %192
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v861 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v856 + (v857 + v751 * v858), v859, v860);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v861, v852);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      };
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_S, PIPE_V, EVENT_ID1);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      // pto: %193, %194
      ;
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      if ((int64_t) ((uint64_t) v524 - (uint64_t) v693) == v65) {
        // pto: %195
        ;
        int64_t v862 = (int64_t) ((uint64_t) v522 + (uint64_t) v693);
        int64_t v863 = (int64_t) ((uint64_t) v692 * (uint64_t) v70);
        // pto: %196
        ;
        float v864 = v638.GetValue(v863);
        // pto: %199
        ;
        float v865 = v864 + v80;
        // pto: %202, %200
        ;
        float v866 = v638.GetValue((int64_t) ((uint64_t) v863 + (uint64_t) v65));
        // pto: %203
        ;
        float v867 = v866 + v80;
        // pto: %206, %204
        ;
        float v868 = v638.GetValue((int64_t) ((uint64_t) v863 + (uint64_t) v75));
        // pto: %207
        ;
        float v869 = v868 + v80;
        // pto: %210, %208
        ;
        float v870 = v638.GetValue((int64_t) ((uint64_t) v863 + (uint64_t) v68));
        // pto: %211
        ;
        float v871 = v870 + v80;
        // pto: %16
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v872 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %16
        ;
        uint64_t v873 = (uint64_t) v56;
        TASSIGN(v872, v873);
        // pto: %212
        ;
        int64_t v874 = v862 < v74 ? v74 : v862;
        // pto: %213
        ;
        __gm__ bfloat16_t* v875 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %213
        ;
        const int64_t v876 = 0;
        // pto: %213
        ;
        const int64_t v877 = 16384;
        // pto: %213
        ;
        pto::Shape<1, 1, 1, 1, 4096> v878 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %213
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v879 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %213
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v880 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v875 + (v876 + v874 * v877), v878, v879);
        TLOAD(v872, v880);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %17
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v881 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %17
        ;
        uint64_t v882 = (uint64_t) v34;
        TASSIGN(v881, v882);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v883 = RoundMode::CAST_ROUND;
        SaturationMode v884 = SaturationMode::ON;
        TCVT(v881, v872, v883, v884);
        // pto: %18
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v885 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %18
        ;
        uint64_t v886 = (uint64_t) v34;
        TASSIGN(v885, v886);
        pipe_barrier(PIPE_V);
        TMULS(v885, v881, v865);
        // pto: %19
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v887 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %19
        ;
        uint64_t v888 = (uint64_t) v57;
        TASSIGN(v887, v888);
        // pto: %215
        ;
        __gm__ bfloat16_t* v889 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %215
        ;
        const int64_t v890 = 4096;
        // pto: %215
        ;
        const int64_t v891 = 16384;
        // pto: %215
        ;
        pto::Shape<1, 1, 1, 1, 4096> v892 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %215
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v893 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %215
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v894 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v889 + (v890 + v874 * v891), v892, v893);
        TLOAD(v887, v894);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %20
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v895 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %20
        ;
        uint64_t v896 = (uint64_t) v56;
        TASSIGN(v895, v896);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v897 = RoundMode::CAST_ROUND;
        SaturationMode v898 = SaturationMode::ON;
        TCVT(v895, v887, v897, v898);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %21
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v899 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %21
        ;
        uint64_t v900 = (uint64_t) v56;
        TASSIGN(v899, v900);
        pipe_barrier(PIPE_V);
        TMULS(v899, v895, v867);
        // pto: %22
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v901 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %22
        ;
        uint64_t v902 = (uint64_t) v34;
        TASSIGN(v901, v902);
        pipe_barrier(PIPE_V);
        TADD(v901, v885, v899);
        // pto: %23
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v903 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %23
        ;
        uint64_t v904 = (uint64_t) v57;
        TASSIGN(v903, v904);
        // pto: %217
        ;
        __gm__ bfloat16_t* v905 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %217
        ;
        const int64_t v906 = 8192;
        // pto: %217
        ;
        const int64_t v907 = 16384;
        // pto: %217
        ;
        pto::Shape<1, 1, 1, 1, 4096> v908 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %217
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v909 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %217
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v910 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v905 + (v906 + v874 * v907), v908, v909);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v903, v910);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %24
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v911 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %24
        ;
        uint64_t v912 = (uint64_t) v56;
        TASSIGN(v911, v912);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v913 = RoundMode::CAST_ROUND;
        SaturationMode v914 = SaturationMode::ON;
        TCVT(v911, v903, v913, v914);
        // pto: %25
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v915 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %25
        ;
        uint64_t v916 = (uint64_t) v56;
        TASSIGN(v915, v916);
        pipe_barrier(PIPE_V);
        TMULS(v915, v911, v869);
        // pto: %26
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v917 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %26
        ;
        uint64_t v918 = (uint64_t) v63;
        TASSIGN(v917, v918);
        // pto: %219
        ;
        __gm__ bfloat16_t* v919 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %219
        ;
        const int64_t v920 = 12288;
        // pto: %219
        ;
        const int64_t v921 = 16384;
        // pto: %219
        ;
        pto::Shape<1, 1, 1, 1, 4096> v922 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %219
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v923 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %219
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v924 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v919 + (v920 + v874 * v921), v922, v923);
        TLOAD(v917, v924);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %27
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v925 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %27
        ;
        uint64_t v926 = (uint64_t) v57;
        TASSIGN(v925, v926);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v927 = RoundMode::CAST_ROUND;
        SaturationMode v928 = SaturationMode::ON;
        TCVT(v925, v917, v927, v928);
        // pto: %28
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v929 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %28
        ;
        uint64_t v930 = (uint64_t) v57;
        TASSIGN(v929, v930);
        pipe_barrier(PIPE_V);
        TMULS(v929, v925, v871);
        // pto: %29
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v931 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %29
        ;
        uint64_t v932 = (uint64_t) v56;
        TASSIGN(v931, v932);
        pipe_barrier(PIPE_V);
        TADD(v931, v915, v929);
        // pto: %30
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v933 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %30
        ;
        uint64_t v934 = (uint64_t) v34;
        TASSIGN(v933, v934);
        pipe_barrier(PIPE_V);
        TADD(v933, v901, v931);
        // pto: %31
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v935 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v71);
        // pto: %31
        ;
        uint64_t v936 = (uint64_t) v34;
        TASSIGN(v935, v936);
        pipe_barrier(PIPE_V);
        RoundMode v937 = RoundMode::CAST_RINT;
        SaturationMode v938 = SaturationMode::ON;
        TCVT(v935, v933, v937, v938);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        // pto: %222
        ;
        __gm__ bfloat16_t* v939 = PTOAS__GLOBAL_TENSOR_DATA(v174);
        // pto: %222
        ;
        const int64_t v940 = 0;
        // pto: %222
        ;
        const int64_t v941 = 4096;
        // pto: %222
        ;
        pto::Shape<1, 1, 1, 1, 4096> v942 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %222
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v943 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %222
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v944 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v939 + (v940 + v874 * v941), v942, v943);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        TSTORE(v944, v935);
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