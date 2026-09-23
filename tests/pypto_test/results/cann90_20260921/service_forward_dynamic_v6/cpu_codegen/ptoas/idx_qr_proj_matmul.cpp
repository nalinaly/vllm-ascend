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

AICORE void idx_qr_proj_matmul(__gm__ int32_t* v1, __gm__ int8_t* v2, __gm__ int8_t* v3, int64_t v4, int64_t v5, int64_t v6, int32_t v7, int32_t v8) {
  using T = float;

  #if defined(__DAV_CUBE__)
  // pto: %c0_i64
  const int64_t v9 = 0;
  // pto: %c4096_i64
  const int64_t v10 = 4096;
  // pto: %c135168_i64
  const int64_t v11 = 135168;
  // pto: %c139264_i64
  const int64_t v12 = 139264;
  // pto: %c3072_i64
  const int64_t v13 = 3072;
  // pto: %c32768_i64
  const int64_t v14 = 32768;
  // pto: %c1024_i64
  const int64_t v15 = 1024;
  // pto: %c2048_i64
  const int64_t v16 = 2048;
  // pto: %c384_index
  const int64_t v17 = 384;
  // pto: %c8192_index
  const int64_t v18 = 8192;
  // pto: %c1_index
  const int64_t v19 = 1;
  // pto: %c1024_index
  const int64_t v20 = 1024;
  // pto: %c8_index
  const int64_t v21 = 8;
  // pto: %c24_index
  const int64_t v22 = 24;
  // pto: %c16_index
  const int64_t v23 = 16;
  // pto: %c0_index
  const int64_t v24 = 0;
  // pto: %c512_index
  const int64_t v25 = 512;
  // pto: %c4_index
  const int64_t v26 = 4;
  // pto: %c2_index
  const int64_t v27 = 2;
  // pto: %c256_index
  const int64_t v28 = 256;
  // pto: %c128_index
  const int64_t v29 = 128;
  // pto: %c64_index
  const int64_t v30 = 64;
  // pto: %cn64_index
  const int64_t v31 = -64;
  // pto: %qr_acc_pad_inline922_inline2319__ssa_v0_view
  const int64_t v32 = 1;
  // pto: %qr_acc_pad_inline922_inline2319__ssa_v0_view
  const int64_t v33 = 1;
  // pto: %qr_acc_pad_inline922_inline2319__ssa_v0_view
  const int64_t v34 = 1;
  // pto: %qr_acc_pad_inline922_inline2319__ssa_v0_view
  int64_t v35 = v17 * v18;
  // pto: %qr_acc_pad_inline922_inline2319__ssa_v0_view
  int64_t v36 = v34 * v35;
  // pto: %qr_acc_pad_inline922_inline2319__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v37 = pto::Shape<1, 1, 1, -1, -1>(v32, v33, v34, v17, v18);
  // pto: %qr_acc_pad_inline922_inline2319__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v38 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v36, v36, v35, v18, v19);
  // pto: %qr_acc_pad_inline922_inline2319__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v39 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v37, v38);
  // pto: %qr__ssa_v0_view
  const int64_t v40 = 1;
  // pto: %qr__ssa_v0_view
  const int64_t v41 = 1;
  // pto: %qr__ssa_v0_view
  const int64_t v42 = 1;
  // pto: %qr__ssa_v0_view
  int64_t v43 = (int64_t) v6;
  // pto: %qr__ssa_v0_view
  int64_t v44 = v43 * v20;
  // pto: %qr__ssa_v0_view
  int64_t v45 = v42 * v44;
  // pto: %qr__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v46 = pto::Shape<1, 1, 1, -1, -1>(v40, v41, v42, v43, v20);
  // pto: %qr__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v47 = pto::Stride<-1, -1, -1, -1, -1>(v41 * v45, v45, v44, v20, v19);
  // pto: %qr__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v48 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v46, v47);
  // pto: %idx_wq_b__ssa_v0_view
  const int64_t v49 = 1;
  // pto: %idx_wq_b__ssa_v0_view
  const int64_t v50 = 1;
  // pto: %idx_wq_b__ssa_v0_view
  const int64_t v51 = 1;
  // pto: %idx_wq_b__ssa_v0_view
  int64_t v52 = v20 * v18;
  // pto: %idx_wq_b__ssa_v0_view
  int64_t v53 = v51 * v52;
  // pto: %idx_wq_b__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v54 = pto::Shape<1, 1, 1, -1, -1>(v49, v50, v51, v20, v18);
  // pto: %idx_wq_b__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v55 = pto::Stride<-1, -1, -1, -1, -1>(v50 * v53, v53, v52, v18, v19);
  // pto: %idx_wq_b__ssa_v0_view
  GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v56 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v54, v55);
  // pto: %qr_proj_worker_inline912_inline2320__ssa_v0
  // pto: %11
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
  for (int64_t v57 = (int64_t) v7; v57 < ((int64_t) ((uint64_t) v4 * (uint64_t) v21)); v57 += v22) {
    // pto: %12
    ;
    int64_t v58 = v57 / v21;
    // pto: %15
    ;
    int64_t v59 = (int64_t) ((uint64_t) v58 * (uint64_t) v23);
    // pto: %16
    ;
    int64_t v60 = (int64_t) ((uint64_t) v5 - (uint64_t) v59);
    // pto: %17
    ;
    int64_t v61 = v60 < v23 ? v60 : v23;
    // pto: %14, %13, %18
    ;
    int64_t v62 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v57 - (uint64_t) ((int64_t) (uint64_t) v58 * (uint64_t) v21)) * (uint64_t) v20);
    for (int64_t v63 = v24; v63 < v20; v63 += v25) {
      // pto: %qr_acc_inline899_inline2332__tile
      ;
      Tile<TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v64 = Tile<TileType::Acc, int32_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v23, v25);
      // pto: %qr_acc_inline899_inline2332__tile
      ;
      uint64_t v65 = (uint64_t) v9;
      TASSIGN(v64, v65);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      for (int64_t v66 = v24; v66 < v26; v66 += v27) {
        // pto: %19
        ;
        int64_t v67 = (int64_t) ((uint64_t) v66 * (uint64_t) v28);
        // pto: %21
        ;
        int64_t v68 = (int64_t) ((uint64_t) v67 + (uint64_t) v28);
        // pto: %qr_tile_inline908_inline2307__tile
        ;
        Tile<TileType::Mat, int8_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v69 = Tile<TileType::Mat, int8_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v61, v28);
        // pto: %qr_tile_inline908_inline2307__tile
        ;
        uint64_t v70 = (uint64_t) v9;
        TASSIGN(v69, v70);
        // pto: %22
        ;
        int64_t v71 = v59 < v24 ? v24 : v59;
        // pto: %23
        ;
        int64_t v72 = v67 < v24 ? v24 : v67;
        // pto: %qr__ssa_v0_pview
        ;
        const int64_t v73 = 0;
        // pto: %qr__ssa_v0_pview
        ;
        __gm__ int8_t* v74 = PTOAS__GLOBAL_TENSOR_DATA(v48);
        // pto: %qr__ssa_v0_pview
        ;
        const int64_t v75 = 1;
        // pto: %qr__ssa_v0_pview
        ;
        const int64_t v76 = 1;
        // pto: %qr__ssa_v0_pview
        ;
        const int64_t v77 = 1;
        // pto: %qr__ssa_v0_pview
        ;
        int64_t v78 = v61 * v20;
        // pto: %qr__ssa_v0_pview
        ;
        int64_t v79 = v77 * v78;
        // pto: %qr__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, -1, 256> v80 = pto::Shape<1, 1, 1, -1, 256>(v75, v76, v77, v61, v28);
        // pto: %qr__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v81 = pto::Stride<-1, -1, -1, -1, -1>(v76 * v79, v79, v78, v20, v19);
        // pto: %qr__ssa_v0_pview
        ;
        GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v82 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v74 + (v73 + v71 * v20 + v72 * v19), v80, v81);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        TLOAD(v69, v82);
        // pto: %wq_tile_inline927_inline2310__tile
        ;
        Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v83 = Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v28, v25);
        // pto: %wq_tile_inline927_inline2310__tile
        ;
        uint64_t v84 = (uint64_t) v10;
        TASSIGN(v83, v84);
        // pto: %25
        ;
        int64_t v85 = (int64_t) ((uint64_t) v62 + (uint64_t) v63);
        // pto: %26
        ;
        int64_t v86 = v85 < v24 ? v24 : v85;
        // pto: %idx_wq_b__ssa_v0_pview
        ;
        __gm__ int8_t* v87 = PTOAS__GLOBAL_TENSOR_DATA(v56);
        // pto: %idx_wq_b__ssa_v0_pview
        ;
        const int64_t v88 = 0;
        // pto: %idx_wq_b__ssa_v0_pview
        ;
        const int64_t v89 = 8192;
        // pto: %idx_wq_b__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 256, 512> v90 = pto::Shape<1, 1, 1, 256, 512>();
        // pto: %idx_wq_b__ssa_v0_pview
        ;
        pto::Stride<2097152, 2097152, 2097152, 8192, 1> v91 = pto::Stride<2097152, 2097152, 2097152, 8192, 1>();
        // pto: %idx_wq_b__ssa_v0_pview
        ;
        GlobalTensor<int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<2097152, 2097152, 2097152, 8192, 1>, pto::Layout::ND> v92 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<2097152, 2097152, 2097152, 8192, 1>, pto::Layout::ND>(v87 + (v88 + v72 * v89 + v86), v90, v91);
        TLOAD(v83, v92);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        // pto: %0
        ;
        Tile<TileType::Mat, int8_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v93 = Tile<TileType::Mat, int8_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v61, v28);
        // pto: %0
        ;
        uint64_t v94 = (uint64_t) v11;
        TASSIGN(v93, v94);
        // pto: %28
        ;
        int64_t v95 = v68 < v24 ? v24 : v68;
        // pto: %29
        ;
        const int64_t v96 = 0;
        // pto: %29
        ;
        __gm__ int8_t* v97 = PTOAS__GLOBAL_TENSOR_DATA(v48);
        // pto: %29
        ;
        const int64_t v98 = 1;
        // pto: %29
        ;
        const int64_t v99 = 1;
        // pto: %29
        ;
        const int64_t v100 = 1;
        // pto: %29
        ;
        int64_t v101 = v61 * v20;
        // pto: %29
        ;
        int64_t v102 = v100 * v101;
        // pto: %29
        ;
        pto::Shape<1, 1, 1, -1, 256> v103 = pto::Shape<1, 1, 1, -1, 256>(v98, v99, v100, v61, v28);
        // pto: %29
        ;
        pto::Stride<-1, -1, -1, -1, -1> v104 = pto::Stride<-1, -1, -1, -1, -1>(v99 * v102, v102, v101, v20, v19);
        // pto: %29
        ;
        GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v105 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v97 + (v96 + v71 * v20 + v95 * v19), v103, v104);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        TLOAD(v93, v105);
        // pto: %1
        ;
        Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v106 = Tile<TileType::Mat, int8_t, 256, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v28, v25);
        // pto: %1
        ;
        uint64_t v107 = (uint64_t) v12;
        TASSIGN(v106, v107);
        // pto: %33
        ;
        __gm__ int8_t* v108 = PTOAS__GLOBAL_TENSOR_DATA(v56);
        // pto: %33
        ;
        const int64_t v109 = 0;
        // pto: %33
        ;
        const int64_t v110 = 8192;
        // pto: %33
        ;
        pto::Shape<1, 1, 1, 256, 512> v111 = pto::Shape<1, 1, 1, 256, 512>();
        // pto: %33
        ;
        pto::Stride<2097152, 2097152, 2097152, 8192, 1> v112 = pto::Stride<2097152, 2097152, 2097152, 8192, 1>();
        // pto: %33
        ;
        GlobalTensor<int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<2097152, 2097152, 2097152, 8192, 1>, pto::Layout::ND> v113 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 256, 512>, pto::Stride<2097152, 2097152, 2097152, 8192, 1>, pto::Layout::ND>(v108 + (v109 + v95 * v110 + v86), v111, v112);
        TLOAD(v106, v113);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        for (int64_t v114 = v24; v114 < v28; v114 += v29) {
          // pto: %qr_acc_inline899_inline2332__tile_l0_a
          ;
          Tile<TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v115 = Tile<TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v61, v30);
          // pto: %qr_acc_inline899_inline2332__tile_l0_a
          ;
          uint64_t v116 = (uint64_t) v13;
          TASSIGN(v115, v116);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          pipe_barrier(PIPE_MTE1);
          TEXTRACT(v115, v69, v24, v114);
          // pto: %qr_acc_inline899_inline2332__tile_l0_b
          ;
          Tile<TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v117 = Tile<TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v30, v25);
          // pto: %qr_acc_inline899_inline2332__tile_l0_b
          ;
          uint64_t v118 = (uint64_t) v14;
          TASSIGN(v117, v118);
          TEXTRACT(v117, v83, v114, v24);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          // pto: %2
          ;
          Tile<TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v119 = Tile<TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v61, v30);
          // pto: %2
          ;
          uint64_t v120 = (uint64_t) v9;
          TASSIGN(v119, v120);
          // pto: %34
          ;
          int64_t v121 = (int64_t) ((uint64_t) v114 + (uint64_t) v30);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
          TEXTRACT(v119, v69, v24, v121);
          // pto: %3
          ;
          Tile<TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v122 = Tile<TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v30, v25);
          // pto: %3
          ;
          uint64_t v123 = (uint64_t) v9;
          TASSIGN(v122, v123);
          TEXTRACT(v122, v83, v121, v24);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          // pto: %36
          ;
          bool v124 = v67 == v24;
          // pto: %37, %38
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          if (v124 & v114 == v24) {
            pipe_barrier(PIPE_M);
            TMATMUL(v64, v115, v117);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v64, v64, v115, v117);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          // pto: %40, %41
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          if (v124 & v114 == v31) {
            pipe_barrier(PIPE_M);
            TMATMUL(v64, v119, v122);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v64, v64, v119, v122);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        };
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        for (int64_t v125 = v24; v125 < v28; v125 += v29) {
          // pto: %5
          ;
          Tile<TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v126 = Tile<TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v61, v30);
          // pto: %5
          ;
          uint64_t v127 = (uint64_t) v15;
          TASSIGN(v126, v127);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
          TEXTRACT(v126, v93, v24, v125);
          // pto: %6
          ;
          Tile<TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v128 = Tile<TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v30, v25);
          // pto: %6
          ;
          uint64_t v129 = (uint64_t) v14;
          TASSIGN(v128, v129);
          pipe_barrier(PIPE_MTE1);
          TEXTRACT(v128, v106, v125, v24);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
          // pto: %7
          ;
          Tile<TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v130 = Tile<TileType::Left, int8_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v61, v30);
          // pto: %7
          ;
          uint64_t v131 = (uint64_t) v16;
          TASSIGN(v130, v131);
          // pto: %43
          ;
          int64_t v132 = (int64_t) ((uint64_t) v125 + (uint64_t) v30);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
          TEXTRACT(v130, v93, v24, v132);
          // pto: %8
          ;
          Tile<TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v133 = Tile<TileType::Right, int8_t, 64, 512, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v30, v25);
          // pto: %8
          ;
          uint64_t v134 = (uint64_t) v9;
          TASSIGN(v133, v134);
          TEXTRACT(v133, v106, v132, v24);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
          // pto: %45
          ;
          bool v135 = v68 == v24;
          // pto: %46, %47
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
          if (v135 & v125 == v24) {
            pipe_barrier(PIPE_M);
            TMATMUL(v64, v126, v128);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v64, v64, v126, v128);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
          // pto: %49, %50
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
          if (v135 & v125 == v31) {
            pipe_barrier(PIPE_M);
            TMATMUL(v64, v130, v133);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v64, v64, v130, v133);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
        };
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      };
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      // pto: %51
      ;
      int64_t v136 = v59 < v24 ? v24 : v59;
      // pto: %52
      ;
      int64_t v137 = (int64_t) ((uint64_t) v62 + (uint64_t) v63);
      // pto: %53
      ;
      int64_t v138 = v137 < v24 ? v24 : v137;
      // pto: %qr_acc_pad_inline922_inline2319__iter_v3_pview
      ;
      __gm__ int32_t* v139 = PTOAS__GLOBAL_TENSOR_DATA(v39);
      // pto: %qr_acc_pad_inline922_inline2319__iter_v3_pview
      ;
      const int64_t v140 = 0;
      // pto: %qr_acc_pad_inline922_inline2319__iter_v3_pview
      ;
      const int64_t v141 = 8192;
      // pto: %qr_acc_pad_inline922_inline2319__iter_v3_pview
      ;
      pto::Shape<1, 1, 1, 16, 512> v142 = pto::Shape<1, 1, 1, 16, 512>();
      // pto: %qr_acc_pad_inline922_inline2319__iter_v3_pview
      ;
      pto::Stride<131072, 131072, 131072, 8192, 1> v143 = pto::Stride<131072, 131072, 131072, 8192, 1>();
      // pto: %qr_acc_pad_inline922_inline2319__iter_v3_pview
      ;
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<131072, 131072, 131072, 8192, 1>, pto::Layout::ND> v144 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<131072, 131072, 131072, 8192, 1>, pto::Layout::ND>(v139 + (v140 + v136 * v141 + v138), v142, v143);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      pipe_barrier(PIPE_FIX);
      TSTORE(v144, v64);
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    };
  }
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}