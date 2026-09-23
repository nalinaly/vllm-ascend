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

AICORE void kv_score_proj(__gm__ float* v1, __gm__ float* v2, __gm__ bfloat16_t* v3, __gm__ bfloat16_t* v4, __gm__ bfloat16_t* v5, int64_t v6, int64_t v7, int64_t v8, int32_t v9, int32_t v10) {
  using T = float;

  #if defined(__DAV_CUBE__)
  // pto: %c0_i64
  const int64_t v11 = 0;
  // pto: %c8192_i64
  const int64_t v12 = 8192;
  // pto: %c49152_i64
  const int64_t v13 = 49152;
  // pto: %c16384_i64
  const int64_t v14 = 16384;
  // pto: %c32768_i64
  const int64_t v15 = 32768;
  // pto: %c40960_i64
  const int64_t v16 = 40960;
  // pto: %c384_index
  const int64_t v17 = 384;
  // pto: %c1024_index
  const int64_t v18 = 1024;
  // pto: %c1_index
  const int64_t v19 = 1;
  // pto: %c4096_index
  const int64_t v20 = 4096;
  // pto: %c2_index
  const int64_t v21 = 2;
  // pto: %c24_index
  const int64_t v22 = 24;
  // pto: %c32_index
  const int64_t v23 = 32;
  // pto: %c64_index
  const int64_t v24 = 64;
  // pto: %c0_index
  const int64_t v25 = 0;
  // pto: %c128_index
  const int64_t v26 = 128;
  // pto: %c512_index
  const int64_t v27 = 512;
  // pto: %c256_index
  const int64_t v28 = 256;
  // pto: %cmp4_kv_proj_pad_inline490_inline1996__ssa_v0_view
  const int64_t v29 = 1;
  // pto: %cmp4_kv_proj_pad_inline490_inline1996__ssa_v0_view
  const int64_t v30 = 1;
  // pto: %cmp4_kv_proj_pad_inline490_inline1996__ssa_v0_view
  const int64_t v31 = 1;
  // pto: %cmp4_kv_proj_pad_inline490_inline1996__ssa_v0_view
  int64_t v32 = v17 * v18;
  // pto: %cmp4_kv_proj_pad_inline490_inline1996__ssa_v0_view
  int64_t v33 = v31 * v32;
  // pto: %cmp4_kv_proj_pad_inline490_inline1996__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v34 = pto::Shape<1, 1, 1, -1, -1>(v29, v30, v31, v17, v18);
  // pto: %cmp4_kv_proj_pad_inline490_inline1996__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v35 = pto::Stride<-1, -1, -1, -1, -1>(v30 * v33, v33, v32, v18, v19);
  // pto: %cmp4_kv_proj_pad_inline490_inline1996__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v36 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v34, v35);
  // pto: %cmp4_score_proj_pad_inline493_inline1987__ssa_v0_view
  const int64_t v37 = 1;
  // pto: %cmp4_score_proj_pad_inline493_inline1987__ssa_v0_view
  const int64_t v38 = 1;
  // pto: %cmp4_score_proj_pad_inline493_inline1987__ssa_v0_view
  const int64_t v39 = 1;
  // pto: %cmp4_score_proj_pad_inline493_inline1987__ssa_v0_view
  int64_t v40 = v17 * v18;
  // pto: %cmp4_score_proj_pad_inline493_inline1987__ssa_v0_view
  int64_t v41 = v39 * v40;
  // pto: %cmp4_score_proj_pad_inline493_inline1987__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v42 = pto::Shape<1, 1, 1, -1, -1>(v37, v38, v39, v17, v18);
  // pto: %cmp4_score_proj_pad_inline493_inline1987__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v43 = pto::Stride<-1, -1, -1, -1, -1>(v38 * v41, v41, v40, v18, v19);
  // pto: %cmp4_score_proj_pad_inline493_inline1987__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v44 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v42, v43);
  // pto: %x_flat_inline489_inline1994__ssa_v0_view
  const int64_t v45 = 1;
  // pto: %x_flat_inline489_inline1994__ssa_v0_view
  const int64_t v46 = 1;
  // pto: %x_flat_inline489_inline1994__ssa_v0_view
  const int64_t v47 = 1;
  // pto: %x_flat_inline489_inline1994__ssa_v0_view
  int64_t v48 = (int64_t) v8;
  // pto: %x_flat_inline489_inline1994__ssa_v0_view
  int64_t v49 = v48 * v20;
  // pto: %x_flat_inline489_inline1994__ssa_v0_view
  int64_t v50 = v47 * v49;
  // pto: %x_flat_inline489_inline1994__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v51 = pto::Shape<1, 1, 1, -1, -1>(v45, v46, v47, v48, v20);
  // pto: %x_flat_inline489_inline1994__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v52 = pto::Stride<-1, -1, -1, -1, -1>(v46 * v50, v50, v49, v20, v19);
  // pto: %x_flat_inline489_inline1994__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v53 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v51, v52);
  // pto: %cmp_wkv__ssa_v0_view
  const int64_t v54 = 1;
  // pto: %cmp_wkv__ssa_v0_view
  const int64_t v55 = 1;
  // pto: %cmp_wkv__ssa_v0_view
  const int64_t v56 = 1;
  // pto: %cmp_wkv__ssa_v0_view
  int64_t v57 = v18 * v20;
  // pto: %cmp_wkv__ssa_v0_view
  int64_t v58 = v56 * v57;
  // pto: %cmp_wkv__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v59 = pto::Shape<1, 1, 1, -1, -1>(v54, v55, v56, v18, v20);
  // pto: %cmp_wkv__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v60 = pto::Stride<-1, -1, -1, -1, -1>(v55 * v58, v58, v57, v20, v19);
  // pto: %cmp_wkv__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v61 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v59, v60);
  // pto: %cmp_wgate__ssa_v0_view
  const int64_t v62 = 1;
  // pto: %cmp_wgate__ssa_v0_view
  const int64_t v63 = 1;
  // pto: %cmp_wgate__ssa_v0_view
  const int64_t v64 = 1;
  // pto: %cmp_wgate__ssa_v0_view
  int64_t v65 = v18 * v20;
  // pto: %cmp_wgate__ssa_v0_view
  int64_t v66 = v64 * v65;
  // pto: %cmp_wgate__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v67 = pto::Shape<1, 1, 1, -1, -1>(v62, v63, v64, v18, v20);
  // pto: %cmp_wgate__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v68 = pto::Stride<-1, -1, -1, -1, -1>(v63 * v66, v66, v65, v20, v19);
  // pto: %cmp_wgate__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v69 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v5, v67, v68);
  // pto: %kv_acc_inline492_inline2011__phi_v5
  Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal> v70 = Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal>(v24, v23);
  // pto: %kv_acc_inline492_inline2011__phi_v5
  uint64_t v71 = (uint64_t) v11;
  TASSIGN(v70, v71);
  // pto: %score_acc_inline499_inline2015__phi_v5
  Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal> v72 = Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal>(v24, v23);
  // pto: %score_acc_inline499_inline2015__phi_v5
  uint64_t v73 = (uint64_t) v12;
  TASSIGN(v72, v73);
  // pto: %kv_worker_inline496_inline2027__ssa_v0
  // pto: %19
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  for (int64_t v74 = (int64_t) v9; v74 < (v6 / v21); v74 += v22) {
    // pto: %20, %21
    ;
    int64_t v75 = (int64_t) ((uint64_t) (v74 / v23) * (uint64_t) v24);
    // pto: %22, %23
    ;
    int64_t v76 = (int64_t) ((uint64_t) (v74 % v23) * (uint64_t) v23);
    // pto: %24
    ;
    int64_t v77 = (int64_t) ((uint64_t) v7 - (uint64_t) v75);
    // pto: %25
    ;
    int64_t v78 = v77 < v24 ? v77 : v24;
    // pto: %kv_acc_inline492_inline2011__tile
    ;
    Tile<TileType::Vec, float, 64, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v79 = Tile<TileType::Vec, float, 64, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v23);
    // pto: %kv_acc_inline492_inline2011__tile
    ;
    uint64_t v80 = (uint64_t) v11;
    TASSIGN(v79, v80);
    // pto: %score_acc_inline499_inline2015__tile
    ;
    Tile<TileType::Vec, float, 64, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v81 = Tile<TileType::Vec, float, 64, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v23);
    // pto: %score_acc_inline499_inline2015__tile
    ;
    uint64_t v82 = (uint64_t) v11;
    TASSIGN(v81, v82);
    // pto: %kv_acc_inline492_inline2011__tile_narrowed_storage
    ;
    Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal> v83 = Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal>(v24, v23);
    // pto: %kv_acc_inline492_inline2011__tile_narrowed_storage
    ;
    uint64_t v84 = (uint64_t) v11;
    TASSIGN(v83, v84);
    v83.SetValidShape(v78, v23);
    // pto: %score_acc_inline499_inline2015__tile_narrowed_storage
    ;
    Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal> v85 = Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal>(v24, v23);
    // pto: %score_acc_inline499_inline2015__tile_narrowed_storage
    ;
    uint64_t v86 = (uint64_t) v12;
    TASSIGN(v85, v86);
    v85.SetValidShape(v78, v23);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int64_t v87 = v25; v87 < v23; v87 += v21) {
      // pto: %26, %30, %27, %28, %29
      ;
      int64_t v88 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v87 * (uint64_t) v26) + (uint64_t) ((int64_t) (uint64_t) (v76 % v27 / v23) * (uint64_t) v28));
      // pto: %31
      ;
      int64_t v89 = v88 % v20;
      // pto: %37, %38
      ;
      int64_t v90 = (int64_t) ((uint64_t) v88 + (uint64_t) v26) % v20;
      // pto: %x_tile_inline497_inline2013__tile
      ;
      Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v91 = Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v78, v26);
      // pto: %x_tile_inline497_inline2013__tile
      ;
      uint64_t v92 = (uint64_t) v13;
      TASSIGN(v91, v92);
      // pto: %39
      ;
      int64_t v93 = v75 < v25 ? v25 : v75;
      // pto: %40
      ;
      int64_t v94 = v89 < v25 ? v25 : v89;
      // pto: %x_flat_inline489_inline1994__ssa_v0_pview
      ;
      const int64_t v95 = 0;
      // pto: %x_flat_inline489_inline1994__ssa_v0_pview
      ;
      __gm__ bfloat16_t* v96 = PTOAS__GLOBAL_TENSOR_DATA(v53);
      // pto: %x_flat_inline489_inline1994__ssa_v0_pview
      ;
      const int64_t v97 = 1;
      // pto: %x_flat_inline489_inline1994__ssa_v0_pview
      ;
      const int64_t v98 = 1;
      // pto: %x_flat_inline489_inline1994__ssa_v0_pview
      ;
      const int64_t v99 = 1;
      // pto: %x_flat_inline489_inline1994__ssa_v0_pview
      ;
      int64_t v100 = v78 * v20;
      // pto: %x_flat_inline489_inline1994__ssa_v0_pview
      ;
      int64_t v101 = v99 * v100;
      // pto: %x_flat_inline489_inline1994__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, 128> v102 = pto::Shape<1, 1, 1, -1, 128>(v97, v98, v99, v78, v26);
      // pto: %x_flat_inline489_inline1994__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v103 = pto::Stride<-1, -1, -1, -1, -1>(v98 * v101, v101, v100, v20, v19);
      // pto: %x_flat_inline489_inline1994__ssa_v0_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 128>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v104 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 128>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v96 + (v95 + v93 * v20 + v94 * v19), v102, v103);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v91, v104);
      // pto: %wkv_tile_inline483_inline2021__tile
      ;
      Tile<TileType::Mat, bfloat16_t, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v105 = Tile<TileType::Mat, bfloat16_t, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v23, v26);
      // pto: %wkv_tile_inline483_inline2021__tile
      ;
      uint64_t v106 = (uint64_t) v11;
      TASSIGN(v105, v106);
      // pto: %41
      ;
      int64_t v107 = v76 < v25 ? v25 : v76;
      // pto: %cmp_wkv__ssa_v0_pview
      ;
      __gm__ bfloat16_t* v108 = PTOAS__GLOBAL_TENSOR_DATA(v61);
      // pto: %cmp_wkv__ssa_v0_pview
      ;
      const int64_t v109 = 0;
      // pto: %cmp_wkv__ssa_v0_pview
      ;
      const int64_t v110 = 4096;
      // pto: %cmp_wkv__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 32, 128> v111 = pto::Shape<1, 1, 1, 32, 128>();
      // pto: %cmp_wkv__ssa_v0_pview
      ;
      pto::Stride<131072, 131072, 131072, 4096, 1> v112 = pto::Stride<131072, 131072, 131072, 4096, 1>();
      // pto: %cmp_wkv__ssa_v0_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<131072, 131072, 131072, 4096, 1>, pto::Layout::ND> v113 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<131072, 131072, 131072, 4096, 1>, pto::Layout::ND>(v108 + (v109 + v107 * v110 + v94), v111, v112);
      TLOAD(v105, v113);
      // pto: %wgate_tile_inline491_inline1979__tile
      ;
      Tile<TileType::Mat, bfloat16_t, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v114 = Tile<TileType::Mat, bfloat16_t, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v23, v26);
      // pto: %wgate_tile_inline491_inline1979__tile
      ;
      uint64_t v115 = (uint64_t) v12;
      TASSIGN(v114, v115);
      // pto: %cmp_wgate__ssa_v0_pview
      ;
      __gm__ bfloat16_t* v116 = PTOAS__GLOBAL_TENSOR_DATA(v69);
      // pto: %cmp_wgate__ssa_v0_pview
      ;
      const int64_t v117 = 0;
      // pto: %cmp_wgate__ssa_v0_pview
      ;
      const int64_t v118 = 4096;
      // pto: %cmp_wgate__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 32, 128> v119 = pto::Shape<1, 1, 1, 32, 128>();
      // pto: %cmp_wgate__ssa_v0_pview
      ;
      pto::Stride<131072, 131072, 131072, 4096, 1> v120 = pto::Stride<131072, 131072, 131072, 4096, 1>();
      // pto: %cmp_wgate__ssa_v0_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<131072, 131072, 131072, 4096, 1>, pto::Layout::ND> v121 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<131072, 131072, 131072, 4096, 1>, pto::Layout::ND>(v116 + (v117 + v107 * v118 + v94), v119, v120);
      TLOAD(v114, v121);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %0
      ;
      Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v122 = Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v78, v26);
      // pto: %0
      ;
      uint64_t v123 = (uint64_t) v14;
      TASSIGN(v122, v123);
      // pto: %46
      ;
      int64_t v124 = v90 < v25 ? v25 : v90;
      // pto: %47
      ;
      const int64_t v125 = 0;
      // pto: %47
      ;
      __gm__ bfloat16_t* v126 = PTOAS__GLOBAL_TENSOR_DATA(v53);
      // pto: %47
      ;
      const int64_t v127 = 1;
      // pto: %47
      ;
      const int64_t v128 = 1;
      // pto: %47
      ;
      const int64_t v129 = 1;
      // pto: %47
      ;
      int64_t v130 = v78 * v20;
      // pto: %47
      ;
      int64_t v131 = v129 * v130;
      // pto: %47
      ;
      pto::Shape<1, 1, 1, -1, 128> v132 = pto::Shape<1, 1, 1, -1, 128>(v127, v128, v129, v78, v26);
      // pto: %47
      ;
      pto::Stride<-1, -1, -1, -1, -1> v133 = pto::Stride<-1, -1, -1, -1, -1>(v128 * v131, v131, v130, v20, v19);
      // pto: %47
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 128>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v134 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 128>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v126 + (v125 + v93 * v20 + v124 * v19), v132, v133);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      TLOAD(v122, v134);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      // pto: %1
      ;
      Tile<TileType::Mat, bfloat16_t, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v135 = Tile<TileType::Mat, bfloat16_t, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v23, v26);
      // pto: %1
      ;
      uint64_t v136 = (uint64_t) v15;
      TASSIGN(v135, v136);
      // pto: %50
      ;
      __gm__ bfloat16_t* v137 = PTOAS__GLOBAL_TENSOR_DATA(v61);
      // pto: %50
      ;
      const int64_t v138 = 0;
      // pto: %50
      ;
      const int64_t v139 = 4096;
      // pto: %50
      ;
      pto::Shape<1, 1, 1, 32, 128> v140 = pto::Shape<1, 1, 1, 32, 128>();
      // pto: %50
      ;
      pto::Stride<131072, 131072, 131072, 4096, 1> v141 = pto::Stride<131072, 131072, 131072, 4096, 1>();
      // pto: %50
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<131072, 131072, 131072, 4096, 1>, pto::Layout::ND> v142 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<131072, 131072, 131072, 4096, 1>, pto::Layout::ND>(v137 + (v138 + v107 * v139 + v124), v140, v141);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v135, v142);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
      // pto: %2
      ;
      Tile<TileType::Mat, bfloat16_t, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v143 = Tile<TileType::Mat, bfloat16_t, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v23, v26);
      // pto: %2
      ;
      uint64_t v144 = (uint64_t) v16;
      TASSIGN(v143, v144);
      // pto: %53
      ;
      __gm__ bfloat16_t* v145 = PTOAS__GLOBAL_TENSOR_DATA(v69);
      // pto: %53
      ;
      const int64_t v146 = 0;
      // pto: %53
      ;
      const int64_t v147 = 4096;
      // pto: %53
      ;
      pto::Shape<1, 1, 1, 32, 128> v148 = pto::Shape<1, 1, 1, 32, 128>();
      // pto: %53
      ;
      pto::Stride<131072, 131072, 131072, 4096, 1> v149 = pto::Stride<131072, 131072, 131072, 4096, 1>();
      // pto: %53
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<131072, 131072, 131072, 4096, 1>, pto::Layout::ND> v150 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<131072, 131072, 131072, 4096, 1>, pto::Layout::ND>(v145 + (v146 + v107 * v147 + v124), v148, v149);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
      TLOAD(v143, v150);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
      // pto: %54
      ;
      v70.SetValidShape(v78, v23);
      v72.SetValidShape(v78, v23);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      if (v87 == v25) {
        // pto: %wkv_tile_inline483_inline2021__tile_t
        ;
        Tile<TileType::Mat, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v151 = Tile<TileType::Mat, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v26, v23);
        // pto: %wkv_tile_inline483_inline2021__tile_t
        ;
        uint64_t v152 = (uint64_t) v11;
        TASSIGN(v151, v152);
        // pto: %x_tile_inline497_inline2013__tile_Left
        ;
        Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v153 = Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v78, v26);
        // pto: %x_tile_inline497_inline2013__tile_Left
        ;
        uint64_t v154 = (uint64_t) v11;
        TASSIGN(v153, v154);
        TMOV(v153, v91);
        // pto: %wkv_tile_inline483_inline2021__tile_t_Right
        ;
        Tile<TileType::Right, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v155 = Tile<TileType::Right, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v26, v23);
        // pto: %wkv_tile_inline483_inline2021__tile_t_Right
        ;
        uint64_t v156 = (uint64_t) v11;
        TASSIGN(v155, v156);
        TMOV(v155, v151);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        // pto: %3
        ;
        Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal> v157 = Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal>(v78, v23);
        // pto: %3
        ;
        uint64_t v158 = (uint64_t) v11;
        TASSIGN(v157, v158);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL(v157, v153, v155);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        // pto: %wgate_tile_inline491_inline1979__tile_t
        ;
        Tile<TileType::Mat, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v159 = Tile<TileType::Mat, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v26, v23);
        // pto: %wgate_tile_inline491_inline1979__tile_t
        ;
        uint64_t v160 = (uint64_t) v12;
        TASSIGN(v159, v160);
        // pto: %wgate_tile_inline491_inline1979__tile_t_Right
        ;
        Tile<TileType::Right, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v161 = Tile<TileType::Right, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v26, v23);
        // pto: %wgate_tile_inline491_inline1979__tile_t_Right
        ;
        uint64_t v162 = (uint64_t) v11;
        TASSIGN(v161, v162);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        TMOV(v161, v159);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        // pto: %4
        ;
        Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal> v163 = Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal>(v78, v23);
        // pto: %4
        ;
        uint64_t v164 = (uint64_t) v12;
        TASSIGN(v163, v164);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        TMATMUL(v163, v153, v161);
      } else {
        // pto: %5
        ;
        Tile<TileType::Mat, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v165 = Tile<TileType::Mat, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v26, v23);
        // pto: %5
        ;
        uint64_t v166 = (uint64_t) v11;
        TASSIGN(v165, v166);
        // pto: %6
        ;
        Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v167 = Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v78, v26);
        // pto: %6
        ;
        uint64_t v168 = (uint64_t) v11;
        TASSIGN(v167, v168);
        TMOV(v167, v91);
        // pto: %7
        ;
        Tile<TileType::Right, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v169 = Tile<TileType::Right, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v26, v23);
        // pto: %7
        ;
        uint64_t v170 = (uint64_t) v11;
        TASSIGN(v169, v170);
        TMOV(v169, v165);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        TMATMUL_ACC(v83, v83, v167, v169);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        // pto: %9
        ;
        Tile<TileType::Mat, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v171 = Tile<TileType::Mat, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v26, v23);
        // pto: %9
        ;
        uint64_t v172 = (uint64_t) v12;
        TASSIGN(v171, v172);
        // pto: %10
        ;
        Tile<TileType::Right, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v173 = Tile<TileType::Right, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v26, v23);
        // pto: %10
        ;
        uint64_t v174 = (uint64_t) v11;
        TASSIGN(v173, v174);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TMOV(v173, v171);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
        TMATMUL_ACC(v85, v85, v167, v173);
      };
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      // pto: %12
      ;
      Tile<TileType::Mat, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v175 = Tile<TileType::Mat, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v26, v23);
      // pto: %12
      ;
      uint64_t v176 = (uint64_t) v15;
      TASSIGN(v175, v176);
      // pto: %13
      ;
      Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v177 = Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v78, v26);
      // pto: %13
      ;
      uint64_t v178 = (uint64_t) v14;
      TASSIGN(v177, v178);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v177, v122);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      // pto: %14
      ;
      Tile<TileType::Right, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v179 = Tile<TileType::Right, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v26, v23);
      // pto: %14
      ;
      uint64_t v180 = (uint64_t) v12;
      TASSIGN(v179, v180);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
      TMOV(v179, v175);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
      TMATMUL_ACC(v70, v70, v177, v179);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      // pto: %16
      ;
      Tile<TileType::Mat, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v181 = Tile<TileType::Mat, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v26, v23);
      // pto: %16
      ;
      uint64_t v182 = (uint64_t) v16;
      TASSIGN(v181, v182);
      // pto: %17
      ;
      Tile<TileType::Right, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v183 = Tile<TileType::Right, bfloat16_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v26, v23);
      // pto: %17
      ;
      uint64_t v184 = (uint64_t) v12;
      TASSIGN(v183, v184);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
      TMOV(v183, v181);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
      TMATMUL_ACC(v72, v72, v177, v183);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    // pto: %55
    ;
    int64_t v185 = v75 < v25 ? v25 : v75;
    // pto: %56
    ;
    int64_t v186 = v76 < v25 ? v25 : v76;
    // pto: %cmp4_kv_proj_pad_inline490_inline1996__iter_v1_pview
    ;
    const int64_t v187 = 0;
    // pto: %cmp4_kv_proj_pad_inline490_inline1996__iter_v1_pview
    ;
    __gm__ float* v188 = PTOAS__GLOBAL_TENSOR_DATA(v36);
    // pto: %cmp4_kv_proj_pad_inline490_inline1996__iter_v1_pview
    ;
    const int64_t v189 = 1;
    // pto: %cmp4_kv_proj_pad_inline490_inline1996__iter_v1_pview
    ;
    const int64_t v190 = 1;
    // pto: %cmp4_kv_proj_pad_inline490_inline1996__iter_v1_pview
    ;
    const int64_t v191 = 1;
    // pto: %cmp4_kv_proj_pad_inline490_inline1996__iter_v1_pview
    ;
    int64_t v192 = v78 * v18;
    // pto: %cmp4_kv_proj_pad_inline490_inline1996__iter_v1_pview
    ;
    int64_t v193 = v191 * v192;
    // pto: %cmp4_kv_proj_pad_inline490_inline1996__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, -1, 32> v194 = pto::Shape<1, 1, 1, -1, 32>(v189, v190, v191, v78, v23);
    // pto: %cmp4_kv_proj_pad_inline490_inline1996__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v195 = pto::Stride<-1, -1, -1, -1, -1>(v190 * v193, v193, v192, v18, v19);
    // pto: %cmp4_kv_proj_pad_inline490_inline1996__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 32>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v196 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 32>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v188 + (v187 + v185 * v18 + v186 * v19), v194, v195);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v196, v83);
    // pto: %cmp4_score_proj_pad_inline493_inline1987__iter_v1_pview
    ;
    const int64_t v197 = 0;
    // pto: %cmp4_score_proj_pad_inline493_inline1987__iter_v1_pview
    ;
    __gm__ float* v198 = PTOAS__GLOBAL_TENSOR_DATA(v44);
    // pto: %cmp4_score_proj_pad_inline493_inline1987__iter_v1_pview
    ;
    const int64_t v199 = 1;
    // pto: %cmp4_score_proj_pad_inline493_inline1987__iter_v1_pview
    ;
    const int64_t v200 = 1;
    // pto: %cmp4_score_proj_pad_inline493_inline1987__iter_v1_pview
    ;
    const int64_t v201 = 1;
    // pto: %cmp4_score_proj_pad_inline493_inline1987__iter_v1_pview
    ;
    int64_t v202 = v78 * v18;
    // pto: %cmp4_score_proj_pad_inline493_inline1987__iter_v1_pview
    ;
    int64_t v203 = v201 * v202;
    // pto: %cmp4_score_proj_pad_inline493_inline1987__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, -1, 32> v204 = pto::Shape<1, 1, 1, -1, 32>(v199, v200, v201, v78, v23);
    // pto: %cmp4_score_proj_pad_inline493_inline1987__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v205 = pto::Stride<-1, -1, -1, -1, -1>(v200 * v203, v203, v202, v18, v19);
    // pto: %cmp4_score_proj_pad_inline493_inline1987__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 32>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v206 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 32>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v198 + (v197 + v185 * v18 + v186 * v19), v204, v205);
    TSTORE(v206, v85);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}