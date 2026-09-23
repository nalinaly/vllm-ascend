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

AICORE void weights_proj(__gm__ float* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int64_t v4, int64_t v5, int64_t v6, int32_t v7, int32_t v8) {
  using T = float;

  #if defined(__DAV_CUBE__)
  // pto: %c0_i64
  const int64_t v9 = 0;
  // pto: %c8192_i64
  const int64_t v10 = 8192;
  // pto: %c384_index
  const int64_t v11 = 384;
  // pto: %c64_index
  const int64_t v12 = 64;
  // pto: %c1_index
  const int64_t v13 = 1;
  // pto: %c4096_index
  const int64_t v14 = 4096;
  // pto: %c4_index
  const int64_t v15 = 4;
  // pto: %c8_index
  const int64_t v16 = 8;
  // pto: %c16_index
  const int64_t v17 = 16;
  // pto: %c0_index
  const int64_t v18 = 0;
  // pto: %c24_index
  const int64_t v19 = 24;
  // pto: %c2_index
  const int64_t v20 = 2;
  // pto: %c7_index
  const int64_t v21 = 7;
  // pto: %c48_index
  const int64_t v22 = 48;
  // pto: %c15_index
  const int64_t v23 = 15;
  // pto: %c96_index
  const int64_t v24 = 96;
  // pto: %c144_index
  const int64_t v25 = 144;
  // pto: %c192_index
  const int64_t v26 = 192;
  // pto: %c240_index
  const int64_t v27 = 240;
  // pto: %c5_index
  const int64_t v28 = 5;
  // pto: %c14_index
  const int64_t v29 = 14;
  // pto: %c256_index
  const int64_t v30 = 256;
  // pto: %weights_partial_inline1043_inline2417__ssa_v0_view
  const int64_t v31 = 1;
  // pto: %weights_partial_inline1043_inline2417__ssa_v0_view
  const int64_t v32 = 1;
  // pto: %weights_partial_inline1043_inline2417__ssa_v0_view
  const int64_t v33 = 1;
  // pto: %weights_partial_inline1043_inline2417__ssa_v0_view
  int64_t v34 = v11 * v12;
  // pto: %weights_partial_inline1043_inline2417__ssa_v0_view
  int64_t v35 = v33 * v34;
  // pto: %weights_partial_inline1043_inline2417__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v36 = pto::Shape<1, 1, 1, -1, -1>(v31, v32, v33, v11, v12);
  // pto: %weights_partial_inline1043_inline2417__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v37 = pto::Stride<-1, -1, -1, -1, -1>(v32 * v35, v35, v34, v12, v13);
  // pto: %weights_partial_inline1043_inline2417__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v38 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v36, v37);
  // pto: %x_flat_inline1036_inline2434__ssa_v0_view
  const int64_t v39 = 1;
  // pto: %x_flat_inline1036_inline2434__ssa_v0_view
  const int64_t v40 = 1;
  // pto: %x_flat_inline1036_inline2434__ssa_v0_view
  const int64_t v41 = 1;
  // pto: %x_flat_inline1036_inline2434__ssa_v0_view
  int64_t v42 = (int64_t) v6;
  // pto: %x_flat_inline1036_inline2434__ssa_v0_view
  int64_t v43 = v42 * v14;
  // pto: %x_flat_inline1036_inline2434__ssa_v0_view
  int64_t v44 = v41 * v43;
  // pto: %x_flat_inline1036_inline2434__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v45 = pto::Shape<1, 1, 1, -1, -1>(v39, v40, v41, v42, v14);
  // pto: %x_flat_inline1036_inline2434__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v46 = pto::Stride<-1, -1, -1, -1, -1>(v40 * v44, v44, v43, v14, v13);
  // pto: %x_flat_inline1036_inline2434__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v47 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v45, v46);
  // pto: %weights_proj__ssa_v0_view
  const int64_t v48 = 1;
  // pto: %weights_proj__ssa_v0_view
  const int64_t v49 = 1;
  // pto: %weights_proj__ssa_v0_view
  const int64_t v50 = 1;
  // pto: %weights_proj__ssa_v0_view
  int64_t v51 = v14 * v12;
  // pto: %weights_proj__ssa_v0_view
  int64_t v52 = v50 * v51;
  // pto: %weights_proj__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v53 = pto::Shape<1, 1, 1, -1, -1>(v48, v49, v50, v14, v12);
  // pto: %weights_proj__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v54 = pto::Stride<-1, -1, -1, -1, -1>(v49 * v52, v52, v51, v12, v13);
  // pto: %weights_proj__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v55 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v53, v54);
  // pto: %w_worker_inline1050_inline2387__ssa_v0
  // pto: %1
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (int64_t v56 = (int64_t) v7; v56 < ((int64_t) ((uint64_t) v4 * (uint64_t) v15)); v56 += v16) {
    // pto: %2
    ;
    int64_t v57 = v56 / v15;
    // pto: %3
    ;
    int64_t v58 = v56 % v15;
    // pto: %4
    ;
    int64_t v59 = (int64_t) ((uint64_t) v58 * (uint64_t) v17);
    // pto: %5
    ;
    int64_t v60 = (int64_t) ((uint64_t) v57 * (uint64_t) v17);
    // pto: %6
    ;
    int64_t v61 = (int64_t) ((uint64_t) v5 - (uint64_t) v60);
    // pto: %7
    ;
    int64_t v62 = v61 < v17 ? v61 : v17;
    // pto: %weights_acc_inline1045_inline2411__tile
    ;
    Tile<TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v63 = Tile<TileType::Acc, float, 16, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v17, v17);
    // pto: %weights_acc_inline1045_inline2411__tile
    ;
    uint64_t v64 = (uint64_t) v9;
    TASSIGN(v63, v64);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int64_t v65 = v18; v65 < v17; v65 += v13) {
      // pto: %8
      ;
      // pto: %k_order_inline1033_inline2393__phi_v6
      ;
      int64_t v66;
      if (v5 == v19) {
        // pto: %9
        ;
        int64_t v67 = v65 / v20;
        // pto: %10, %11
        ;
        // pto: %k_direction_inline1049_inline2426__phi_v2
        ;
        int64_t v68;
        if (v58 % v20 == v13) {
          // pto: %13
          ;
          v68 = (int64_t) ((uint64_t) v21 - (uint64_t) v67);
        } else {
          v68 = v67;
        };
        // pto: %16
        ;
        uint64_t v69 = (uint64_t) v68;
        // pto: %14, %15, %16, %17, %18, %20, %19
        ;
        v66 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) ((uint64_t) ((int64_t) (uint64_t) (v58 / v20) * (uint64_t) v15) + v69) % v16) * (uint64_t) v20) + (uint64_t) (v65 % v20));
      } else {
        // pto: %21
        ;
        // pto: %k_order_inline1033_inline2393__phi_v5
        ;
        int64_t v70;
        if (v5 == v22) {
          // pto: %22, %23
          ;
          // pto: %k_direction_inline1049_inline2426__phi_v5
          ;
          int64_t v71;
          if (v58 % v20 == v13) {
            // pto: %24
            ;
            v71 = (int64_t) ((uint64_t) v23 - (uint64_t) v65);
          } else {
            v71 = v65;
          };
          // pto: %27
          ;
          uint64_t v72 = (uint64_t) v71;
          // pto: %25, %26, %27, %28
          ;
          v70 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) (v58 / v20) * (uint64_t) v16) + v72) % v17;
        } else {
          // pto: %30
          ;
          bool v73 = v5 == v25;
          // pto: %32
          ;
          bool v74 = v5 == v26;
          // pto: %34
          ;
          bool v75 = v5 == v27;
          // pto: %29, %31, %33, %35
          ;
          // pto: %k_order_inline1033_inline2393__phi_v4
          ;
          int64_t v76;
          if ((v5 == v24 | v73 | v74) | v75) {
            // pto: %36, %37
            ;
            // pto: %k_direction_inline1049_inline2426__phi_v8
            ;
            int64_t v77;
            if (v57 % v20 == v13) {
              // pto: %38
              ;
              v77 = (int64_t) ((uint64_t) v23 - (uint64_t) v65);
            } else {
              v77 = v65;
            };
            // pto: %39
            ;
            int64_t v78 = v57 / v20;
            // pto: %40
            ;
            // pto: %k_shift_inline1040_inline2425__phi_v6
            ;
            int64_t v79;
            if (v73) {
              // pto: %42, %43, %44
              ;
              v79 = (int64_t) ((uint64_t) (v57 % v16 / v20) * (uint64_t) v15);
            } else {
              // pto: %k_shift_inline1040_inline2425__phi_v5
              ;
              int64_t v80;
              if (v74) {
                // pto: %47
                ;
                v80 = (int64_t) ((uint64_t) v78 * (uint64_t) v20);
              } else {
                // pto: %k_shift_inline1040_inline2425__phi_v4
                ;
                int64_t v81;
                if (v75) {
                  // pto: %49, %50, %51
                  ;
                  v81 = (int64_t) ((uint64_t) (v57 % v29 / v20) * (uint64_t) v20);
                } else {
                  v81 = (int64_t) ((uint64_t) v78 * (uint64_t) v28);
                };
                v80 = v81;
              };
              v79 = v80;
            };
            // pto: %52
            ;
            uint64_t v82 = (uint64_t) v79;
            // pto: %52
            ;
            uint64_t v83 = (uint64_t) v77;
            // pto: %52, %53
            ;
            v76 = (int64_t) (v82 + v83) % v17;
          } else {
            v76 = v65;
          };
          v70 = v76;
        };
        v66 = v70;
      };
      // pto: %54
      ;
      uint64_t v84 = (uint64_t) v66;
      // pto: %54
      ;
      int64_t v85 = (int64_t) (v84 * (uint64_t) v30);
      // pto: %x_tile_inline1037_inline2391__tile
      ;
      Tile<TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v86 = Tile<TileType::Mat, bfloat16_t, 16, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v62, v30);
      // pto: %x_tile_inline1037_inline2391__tile
      ;
      uint64_t v87 = (uint64_t) v9;
      TASSIGN(v86, v87);
      // pto: %55
      ;
      int64_t v88 = v60 < v18 ? v18 : v60;
      // pto: %56
      ;
      int64_t v89 = v85 < v18 ? v18 : v85;
      // pto: %x_flat_inline1036_inline2434__ssa_v0_pview
      ;
      const int64_t v90 = 0;
      // pto: %x_flat_inline1036_inline2434__ssa_v0_pview
      ;
      __gm__ bfloat16_t* v91 = PTOAS__GLOBAL_TENSOR_DATA(v47);
      // pto: %x_flat_inline1036_inline2434__ssa_v0_pview
      ;
      const int64_t v92 = 1;
      // pto: %x_flat_inline1036_inline2434__ssa_v0_pview
      ;
      const int64_t v93 = 1;
      // pto: %x_flat_inline1036_inline2434__ssa_v0_pview
      ;
      const int64_t v94 = 1;
      // pto: %x_flat_inline1036_inline2434__ssa_v0_pview
      ;
      int64_t v95 = v62 * v14;
      // pto: %x_flat_inline1036_inline2434__ssa_v0_pview
      ;
      int64_t v96 = v94 * v95;
      // pto: %x_flat_inline1036_inline2434__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, 256> v97 = pto::Shape<1, 1, 1, -1, 256>(v92, v93, v94, v62, v30);
      // pto: %x_flat_inline1036_inline2434__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v98 = pto::Stride<-1, -1, -1, -1, -1>(v93 * v96, v96, v95, v14, v13);
      // pto: %x_flat_inline1036_inline2434__ssa_v0_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v99 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v91 + (v90 + v88 * v14 + v89 * v13), v97, v98);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v86, v99);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %weights_proj_tile_inline1030_inline2389__tile
      ;
      Tile<TileType::Mat, bfloat16_t, 256, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v100 = Tile<TileType::Mat, bfloat16_t, 256, 16, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v30, v17);
      // pto: %weights_proj_tile_inline1030_inline2389__tile
      ;
      uint64_t v101 = (uint64_t) v10;
      TASSIGN(v100, v101);
      // pto: %58
      ;
      int64_t v102 = v59 < v18 ? v18 : v59;
      // pto: %weights_proj__ssa_v0_pview
      ;
      __gm__ bfloat16_t* v103 = PTOAS__GLOBAL_TENSOR_DATA(v55);
      // pto: %weights_proj__ssa_v0_pview
      ;
      const int64_t v104 = 0;
      // pto: %weights_proj__ssa_v0_pview
      ;
      const int64_t v105 = 64;
      // pto: %weights_proj__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 256, 16> v106 = pto::Shape<1, 1, 1, 256, 16>();
      // pto: %weights_proj__ssa_v0_pview
      ;
      pto::Stride<16384, 16384, 16384, 64, 1> v107 = pto::Stride<16384, 16384, 16384, 64, 1>();
      // pto: %weights_proj__ssa_v0_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 256, 16>, pto::Stride<16384, 16384, 16384, 64, 1>, pto::Layout::ND> v108 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 256, 16>, pto::Stride<16384, 16384, 16384, 64, 1>, pto::Layout::ND>(v103 + (v104 + v89 * v105 + v102), v106, v107);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      TLOAD(v100, v108);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      // pto: %x_tile_inline1037_inline2391__tile_Left
      ;
      Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v109 = Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v62, v30);
      // pto: %x_tile_inline1037_inline2391__tile_Left
      ;
      uint64_t v110 = (uint64_t) v9;
      TASSIGN(v109, v110);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      TMOV(v109, v86);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      // pto: %weights_proj_tile_inline1030_inline2389__tile_Right
      ;
      Tile<TileType::Right, bfloat16_t, 256, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v111 = Tile<TileType::Right, bfloat16_t, 256, 16, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v30, v17);
      // pto: %weights_proj_tile_inline1030_inline2389__tile_Right
      ;
      uint64_t v112 = (uint64_t) v9;
      TASSIGN(v111, v112);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v111, v100);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      // pto: %59
      ;
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v65 == v18) {
        TMATMUL(v63, v109, v111);
      } else {
        TMATMUL_ACC(v63, v63, v109, v111);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    // pto: %60
    ;
    int64_t v113 = v60 < v18 ? v18 : v60;
    // pto: %61
    ;
    int64_t v114 = v59 < v18 ? v18 : v59;
    // pto: %weights_partial_inline1043_inline2417__iter_v1_pview
    ;
    __gm__ float* v115 = PTOAS__GLOBAL_TENSOR_DATA(v38);
    // pto: %weights_partial_inline1043_inline2417__iter_v1_pview
    ;
    const int64_t v116 = 0;
    // pto: %weights_partial_inline1043_inline2417__iter_v1_pview
    ;
    const int64_t v117 = 64;
    // pto: %weights_partial_inline1043_inline2417__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 16, 16> v118 = pto::Shape<1, 1, 1, 16, 16>();
    // pto: %weights_partial_inline1043_inline2417__iter_v1_pview
    ;
    pto::Stride<1024, 1024, 1024, 64, 1> v119 = pto::Stride<1024, 1024, 1024, 64, 1>();
    // pto: %weights_partial_inline1043_inline2417__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v120 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(v115 + (v116 + v113 * v117 + v114), v118, v119);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v120, v63);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}