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

AICORE void qproj_dequant_rms_nope_rope(__gm__ bfloat16_t* v1, __gm__ float* v2, __gm__ float* v3, __gm__ float* v4, __gm__ int32_t* v5, __gm__ int32_t* v6, __gm__ float* v7, int64_t v8, int64_t v9, int64_t v10, int64_t v11, int64_t v12, int64_t v13, int32_t v14, int32_t v15) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int64_t v16 = 256;
  // pto: %c101376_i64
  const int64_t v17 = 101376;
  // pto: %c101408_i64
  const int64_t v18 = 101408;
  // pto: %c103456_i64
  const int64_t v19 = 103456;
  // pto: %c0_i64
  const int64_t v20 = 0;
  // pto: %c2048_i64
  const int64_t v21 = 2048;
  // pto: %c51200_i64
  const int64_t v22 = 51200;
  // pto: %c18432_i64
  const int64_t v23 = 18432;
  // pto: %c84480_i64
  const int64_t v24 = 84480;
  // pto: %c34816_i64
  const int64_t v25 = 34816;
  // pto: %c67584_i64
  const int64_t v26 = 67584;
  const int64_t v27 = 3840;
  // pto: %c67840_i64
  const int64_t v28 = 67840;
  // pto: %c68096_i64
  const int64_t v29 = 68096;
  // pto: %c100864_i64
  const int64_t v30 = 100864;
  const int64_t v31 = 20224;
  // pto: %c101120_i64
  const int64_t v32 = 101120;
  // pto: %c32768_index
  const int64_t v33 = 32768;
  // pto: %c1_index
  const int64_t v34 = 1;
  // pto: %c512_index
  const int64_t v35 = 512;
  // pto: %c64_index
  const int64_t v36 = 64;
  // pto: %c7_index
  const int64_t v37 = 7;
  // pto: %c8_index
  const int64_t v38 = 8;
  // pto: %c16_index
  const int64_t v39 = 16;
  // pto: %c48_index
  const int64_t v40 = 48;
  // pto: %c4_index
  const int64_t v41 = 4;
  // pto: %c0_index
  const int64_t v42 = 0;
  // pto: %c2_index
  const int64_t v43 = 2;
  // pto: %cst_27
  const float v44 = 1.0f;
  // pto: %cst_28
  const float v45 = 0.001953125f;
  // pto: %cst_29
  const float v46 = 9.99999997E-7f;
  // pto: %c448_index
  const int64_t v47 = 448;
  // pto: %c192_index
  const int64_t v48 = 192;
  // pto: %c0_i32
  const int32_t v49 = 0;
  // pto: %cst_33
  const float v50 = 0.5f;
  // pto: %cst_34
  const float v51 = 2.0f;
  // pto: %cst_35
  const float v52 = 64.0f;
  // pto: %q_flat_inline3189__ssa_v0_view
  const int64_t v53 = 1;
  // pto: %q_flat_inline3189__ssa_v0_view
  const int64_t v54 = 1;
  // pto: %q_flat_inline3189__ssa_v0_view
  const int64_t v55 = 1;
  // pto: %q_flat_inline3189__ssa_v0_view
  int64_t v56 = (int64_t) v10;
  // pto: %q_flat_inline3189__ssa_v0_view
  int64_t v57 = v56 * v33;
  // pto: %q_flat_inline3189__ssa_v0_view
  int64_t v58 = v55 * v57;
  // pto: %q_flat_inline3189__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v59 = pto::Shape<1, 1, 1, -1, -1>(v53, v54, v55, v56, v33);
  // pto: %q_flat_inline3189__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v60 = pto::Stride<-1, -1, -1, -1, -1>(v54 * v58, v58, v57, v33, v34);
  // pto: %q_flat_inline3189__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v61 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v59, v60);
  // pto: %qr_scale_pad_store_inline1827__rv_v2_view
  const int64_t v62 = 1;
  // pto: %qr_scale_pad_store_inline1827__rv_v2_view
  const int64_t v63 = 1;
  // pto: %qr_scale_pad_store_inline1827__rv_v2_view
  const int64_t v64 = 1;
  // pto: %qr_scale_pad_store_inline1827__rv_v2_view
  int64_t v65 = v35 * v34;
  // pto: %qr_scale_pad_store_inline1827__rv_v2_view
  int64_t v66 = v64 * v65;
  // pto: %qr_scale_pad_store_inline1827__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v67 = pto::Shape<1, 1, 1, -1, -1>(v62, v63, v64, v35, v34);
  // pto: %qr_scale_pad_store_inline1827__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v68 = pto::Stride<-1, -1, -1, -1, -1>(v63 * v66, v66, v65, v34, v35);
  // pto: %qr_scale_pad_store_inline1827__rv_v2_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v69 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v67, v68);
  // pto: %freqs_cos__ssa_v0_view
  const int64_t v70 = 1;
  // pto: %freqs_cos__ssa_v0_view
  const int64_t v71 = 1;
  // pto: %freqs_cos__ssa_v0_view
  const int64_t v72 = 1;
  // pto: %freqs_cos__ssa_v0_view
  int64_t v73 = (int64_t) v11;
  // pto: %freqs_cos__ssa_v0_view
  int64_t v74 = v73 * v36;
  // pto: %freqs_cos__ssa_v0_view
  int64_t v75 = v72 * v74;
  // pto: %freqs_cos__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v76 = pto::Shape<1, 1, 1, -1, -1>(v70, v71, v72, v73, v36);
  // pto: %freqs_cos__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v77 = pto::Stride<-1, -1, -1, -1, -1>(v71 * v75, v75, v74, v36, v34);
  // pto: %freqs_cos__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v78 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v76, v77);
  // pto: %q_rope_sin_signed_inline197__ssa_v0_view
  const int64_t v79 = 1;
  // pto: %q_rope_sin_signed_inline197__ssa_v0_view
  const int64_t v80 = 1;
  // pto: %q_rope_sin_signed_inline197__ssa_v0_view
  const int64_t v81 = 1;
  // pto: %q_rope_sin_signed_inline197__ssa_v0_view
  int64_t v82 = (int64_t) v12;
  // pto: %q_rope_sin_signed_inline197__ssa_v0_view
  int64_t v83 = v82 * v36;
  // pto: %q_rope_sin_signed_inline197__ssa_v0_view
  int64_t v84 = v81 * v83;
  // pto: %q_rope_sin_signed_inline197__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v85 = pto::Shape<1, 1, 1, -1, -1>(v79, v80, v81, v82, v36);
  // pto: %q_rope_sin_signed_inline197__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v86 = pto::Stride<-1, -1, -1, -1, -1>(v80 * v84, v84, v83, v36, v34);
  // pto: %q_rope_sin_signed_inline197__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v87 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v85, v86);
  // pto: %q_rope_swap_idx_inline196__ssa_v0_view
  const int64_t v88 = 1;
  // pto: %q_rope_swap_idx_inline196__ssa_v0_view
  const int64_t v89 = 1;
  // pto: %q_rope_swap_idx_inline196__ssa_v0_view
  const int64_t v90 = 1;
  // pto: %q_rope_swap_idx_inline196__ssa_v0_view
  int64_t v91 = (int64_t) v12;
  // pto: %q_rope_swap_idx_inline196__ssa_v0_view
  int64_t v92 = v91 * v36;
  // pto: %q_rope_swap_idx_inline196__ssa_v0_view
  int64_t v93 = v90 * v92;
  // pto: %q_rope_swap_idx_inline196__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v94 = pto::Shape<1, 1, 1, -1, -1>(v88, v89, v90, v91, v36);
  // pto: %q_rope_swap_idx_inline196__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v95 = pto::Stride<-1, -1, -1, -1, -1>(v89 * v93, v93, v92, v36, v34);
  // pto: %q_rope_swap_idx_inline196__ssa_v0_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v96 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v5, v94, v95);
  // pto: %q_proj_i32_inline6_inline1833__ssa_v9_view
  const int64_t v97 = 1;
  // pto: %q_proj_i32_inline6_inline1833__ssa_v9_view
  const int64_t v98 = 1;
  // pto: %q_proj_i32_inline6_inline1833__ssa_v9_view
  const int64_t v99 = 1;
  // pto: %q_proj_i32_inline6_inline1833__ssa_v9_view
  int64_t v100 = (int64_t) v13;
  // pto: %q_proj_i32_inline6_inline1833__ssa_v9_view
  int64_t v101 = v100 * v33;
  // pto: %q_proj_i32_inline6_inline1833__ssa_v9_view
  int64_t v102 = v99 * v101;
  // pto: %q_proj_i32_inline6_inline1833__ssa_v9_view
  pto::Shape<1, 1, 1, -1, -1> v103 = pto::Shape<1, 1, 1, -1, -1>(v97, v98, v99, v100, v33);
  // pto: %q_proj_i32_inline6_inline1833__ssa_v9_view
  pto::Stride<-1, -1, -1, -1, -1> v104 = pto::Stride<-1, -1, -1, -1, -1>(v98 * v102, v102, v101, v33, v34);
  // pto: %q_proj_i32_inline6_inline1833__ssa_v9_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v105 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v103, v104);
  // pto: %wq_b_scale__ssa_v0_view
  const int64_t v106 = 1;
  // pto: %wq_b_scale__ssa_v0_view
  const int64_t v107 = 1;
  // pto: %wq_b_scale__ssa_v0_view
  const int64_t v108 = 1;
  // pto: %wq_b_scale__ssa_v0_view
  const int64_t v109 = 1;
  // pto: %wq_b_scale__ssa_v0_view
  int64_t v110 = v33 * v34;
  // pto: %wq_b_scale__ssa_v0_view
  int64_t v111 = v109 * v110;
  // pto: %wq_b_scale__ssa_v0_view
  int64_t v112 = v108 * v111;
  // pto: %wq_b_scale__ssa_v0_view
  pto::Shape<1, 1, 1, 1, -1> v113 = pto::Shape<1, 1, 1, 1, -1>(v106, v107, v108, v109, v33);
  // pto: %wq_b_scale__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v114 = pto::Stride<-1, -1, -1, -1, -1>(v107 * v112, v112, v111, v110, v34);
  // pto: %wq_b_scale__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v115 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v113, v114);
  // pto: %dq_worker_inline3176__ssa_v0
  // pto: %42, %43, %44
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  for (int64_t v116 = (int64_t) v14; v116 < ((int64_t) ((uint64_t) ((int64_t) ((uint64_t) v8 + (uint64_t) v37) / v38) * (uint64_t) v39)); v116 += v40) {
    // pto: %45, %46
    ;
    int64_t v117 = (int64_t) ((uint64_t) (v116 % v39) * (uint64_t) v41);
    // pto: %47, %48
    ;
    int64_t v118 = (int64_t) ((uint64_t) (v116 / v39) * (uint64_t) v38);
    // pto: %49
    ;
    int64_t v119 = (int64_t) ((uint64_t) v9 + (uint64_t) v118);
    // pto: %50, %51
    ;
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    if ((int64_t) ((uint64_t) v118 + (uint64_t) v38) <= v8) {
      // pto: %qr_scale_dq_t_inline3203__tile
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v120 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v34);
      // pto: %qr_scale_dq_t_inline3203__tile
      ;
      uint64_t v121 = (uint64_t) v17;
      TASSIGN(v120, v121);
      // pto: %52
      ;
      int64_t v122 = v118 < v42 ? v42 : v118;
      // pto: %qr_scale_pad_store_inline1827__rv_v2_pview
      ;
      __gm__ float* v123 = PTOAS__GLOBAL_TENSOR_DATA(v69);
      // pto: %qr_scale_pad_store_inline1827__rv_v2_pview
      ;
      const int64_t v124 = 0;
      // pto: %qr_scale_pad_store_inline1827__rv_v2_pview
      ;
      pto::Shape<1, 1, 1, 8, 1> v125 = pto::Shape<1, 1, 1, 8, 1>();
      // pto: %qr_scale_pad_store_inline1827__rv_v2_pview
      ;
      pto::Stride<8, 8, 8, 1, 512> v126 = pto::Stride<8, 8, 8, 1, 512>();
      // pto: %qr_scale_pad_store_inline1827__rv_v2_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<8, 8, 8, 1, 512>, pto::Layout::DN> v127 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<8, 8, 8, 1, 512>, pto::Layout::DN>(v123 + (v124 + v122), v125, v126);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      TLOAD(v120, v127);
      // pto: %q_cos_il_inline3194__tile
      ;
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v128 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
      // pto: %q_cos_il_inline3194__tile
      ;
      uint64_t v129 = (uint64_t) v18;
      TASSIGN(v128, v129);
      // pto: %53
      ;
      int64_t v130 = v119 < v42 ? v42 : v119;
      // pto: %freqs_cos__ssa_v0_pview
      ;
      __gm__ float* v131 = PTOAS__GLOBAL_TENSOR_DATA(v78);
      // pto: %freqs_cos__ssa_v0_pview
      ;
      const int64_t v132 = 0;
      // pto: %freqs_cos__ssa_v0_pview
      ;
      const int64_t v133 = 64;
      // pto: %freqs_cos__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, 64> v134 = pto::Shape<1, 1, 1, 8, 64>();
      // pto: %freqs_cos__ssa_v0_pview
      ;
      pto::Stride<512, 512, 512, 64, 1> v135 = pto::Stride<512, 512, 512, 64, 1>();
      // pto: %freqs_cos__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND> v136 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND>(v131 + (v132 + v130 * v133), v134, v135);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      TLOAD(v128, v136);
      // pto: %q_sin_signed_inline3206__tile
      ;
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v137 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
      // pto: %q_sin_signed_inline3206__tile
      ;
      uint64_t v138 = (uint64_t) v19;
      TASSIGN(v137, v138);
      // pto: %q_rope_sin_signed_inline197__ssa_v0_pview
      ;
      __gm__ float* v139 = PTOAS__GLOBAL_TENSOR_DATA(v87);
      // pto: %q_rope_sin_signed_inline197__ssa_v0_pview
      ;
      const int64_t v140 = 0;
      // pto: %q_rope_sin_signed_inline197__ssa_v0_pview
      ;
      const int64_t v141 = 64;
      // pto: %q_rope_sin_signed_inline197__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, 64> v142 = pto::Shape<1, 1, 1, 8, 64>();
      // pto: %q_rope_sin_signed_inline197__ssa_v0_pview
      ;
      pto::Stride<512, 512, 512, 64, 1> v143 = pto::Stride<512, 512, 512, 64, 1>();
      // pto: %q_rope_sin_signed_inline197__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND> v144 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND>(v139 + (v140 + v130 * v141), v142, v143);
      pipe_barrier(PIPE_ALL);
      TLOAD(v137, v144);
      // pto: %q_swap_idx_inline3184__tile
      ;
      Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v145 = Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
      // pto: %q_swap_idx_inline3184__tile
      ;
      uint64_t v146 = (uint64_t) v20;
      TASSIGN(v145, v146);
      // pto: %q_rope_swap_idx_inline196__ssa_v0_pview
      ;
      __gm__ int32_t* v147 = PTOAS__GLOBAL_TENSOR_DATA(v96);
      // pto: %q_rope_swap_idx_inline196__ssa_v0_pview
      ;
      const int64_t v148 = 0;
      // pto: %q_rope_swap_idx_inline196__ssa_v0_pview
      ;
      const int64_t v149 = 64;
      // pto: %q_rope_swap_idx_inline196__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, 64> v150 = pto::Shape<1, 1, 1, 8, 64>();
      // pto: %q_rope_swap_idx_inline196__ssa_v0_pview
      ;
      pto::Stride<512, 512, 512, 64, 1> v151 = pto::Stride<512, 512, 512, 64, 1>();
      // pto: %q_rope_swap_idx_inline196__ssa_v0_pview
      ;
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND> v152 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND>(v147 + (v148 + v130 * v149), v150, v151);
      TLOAD(v145, v152);
      for (int64_t v153 = v42; v153 < v41; v153 += v43) {
        // pto: %56, %57
        ;
        int64_t v154 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v117 + (uint64_t) v153) * (uint64_t) v35);
        // pto: %59, %58, %60
        ;
        int64_t v155 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v117 + (uint64_t) ((int64_t) (uint64_t) v153 + (uint64_t) v34)) * (uint64_t) v35);
        // pto: %q_head_acc_inline3223__tile
        ;
        Tile<TileType::Vec, int32_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v156 = Tile<TileType::Vec, int32_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %q_head_acc_inline3223__tile
        ;
        uint64_t v157 = (uint64_t) v21;
        TASSIGN(v156, v157);
        // pto: %62
        ;
        int64_t v158 = v154 < v42 ? v42 : v154;
        // pto: %q_proj_i32_inline6_inline1833__ssa_v9_pview
        ;
        __gm__ int32_t* v159 = PTOAS__GLOBAL_TENSOR_DATA(v105);
        // pto: %q_proj_i32_inline6_inline1833__ssa_v9_pview
        ;
        const int64_t v160 = 0;
        // pto: %q_proj_i32_inline6_inline1833__ssa_v9_pview
        ;
        const int64_t v161 = 32768;
        // pto: %q_proj_i32_inline6_inline1833__ssa_v9_pview
        ;
        pto::Shape<1, 1, 1, 8, 512> v162 = pto::Shape<1, 1, 1, 8, 512>();
        // pto: %q_proj_i32_inline6_inline1833__ssa_v9_pview
        ;
        pto::Stride<262144, 262144, 262144, 32768, 1> v163 = pto::Stride<262144, 262144, 262144, 32768, 1>();
        // pto: %q_proj_i32_inline6_inline1833__ssa_v9_pview
        ;
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND> v164 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND>(v159 + (v160 + v122 * v161 + v158), v162, v163);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        TLOAD(v156, v164);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %t__tile
        ;
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v165 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
        // pto: %t__tile
        ;
        uint64_t v166 = (uint64_t) v22;
        TASSIGN(v165, v166);
        // pto: %wq_b_scale__ssa_v0_pview
        ;
        __gm__ float* v167 = PTOAS__GLOBAL_TENSOR_DATA(v115);
        // pto: %wq_b_scale__ssa_v0_pview
        ;
        const int64_t v168 = 0;
        // pto: %wq_b_scale__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, 512> v169 = pto::Shape<1, 1, 1, 1, 512>();
        // pto: %wq_b_scale__ssa_v0_pview
        ;
        pto::Stride<512, 512, 512, 512, 1> v170 = pto::Stride<512, 512, 512, 512, 1>();
        // pto: %wq_b_scale__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v171 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v167 + (v168 + v158), v169, v170);
        TLOAD(v165, v171);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %0
        ;
        Tile<TileType::Vec, int32_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v172 = Tile<TileType::Vec, int32_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %0
        ;
        uint64_t v173 = (uint64_t) v23;
        TASSIGN(v172, v173);
        // pto: %65
        ;
        int64_t v174 = v155 < v42 ? v42 : v155;
        // pto: %66
        ;
        __gm__ int32_t* v175 = PTOAS__GLOBAL_TENSOR_DATA(v105);
        // pto: %66
        ;
        const int64_t v176 = 0;
        // pto: %66
        ;
        const int64_t v177 = 32768;
        // pto: %66
        ;
        pto::Shape<1, 1, 1, 8, 512> v178 = pto::Shape<1, 1, 1, 8, 512>();
        // pto: %66
        ;
        pto::Stride<262144, 262144, 262144, 32768, 1> v179 = pto::Stride<262144, 262144, 262144, 32768, 1>();
        // pto: %66
        ;
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND> v180 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND>(v175 + (v176 + v122 * v177 + v174), v178, v179);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        TLOAD(v172, v180);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %1
        ;
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v181 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
        // pto: %1
        ;
        uint64_t v182 = (uint64_t) v24;
        TASSIGN(v181, v182);
        // pto: %68
        ;
        __gm__ float* v183 = PTOAS__GLOBAL_TENSOR_DATA(v115);
        // pto: %68
        ;
        const int64_t v184 = 0;
        // pto: %68
        ;
        pto::Shape<1, 1, 1, 1, 512> v185 = pto::Shape<1, 1, 1, 1, 512>();
        // pto: %68
        ;
        pto::Stride<512, 512, 512, 512, 1> v186 = pto::Stride<512, 512, 512, 512, 1>();
        // pto: %68
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v187 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v183 + (v184 + v174), v185, v186);
        TLOAD(v181, v187);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %q_head_scale_inline3213__tile
        ;
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v188 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
        // pto: %q_head_scale_inline3213__tile
        ;
        uint64_t v189 = (uint64_t) v22;
        TASSIGN(v188, v189);
        // pto: %q_head_acc_fp32_inline3173__tile
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v190 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %q_head_acc_fp32_inline3173__tile
        ;
        uint64_t v191 = (uint64_t) v21;
        TASSIGN(v190, v191);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v192 = RoundMode::CAST_NONE;
        SaturationMode v193 = SaturationMode::OFF;
        TCVT(v190, v156, v192, v193);
        // pto: %2
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v194 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %2
        ;
        uint64_t v195 = (uint64_t) v25;
        TASSIGN(v194, v195);
        TEXPANDS(v194, v44);
        // pto: %3
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v196 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %3
        ;
        uint64_t v197 = (uint64_t) v25;
        TASSIGN(v196, v197);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v196, v194, v120);
        // pto: %q_head_scale_combined_inline3180__tile
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v198 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %q_head_scale_combined_inline3180__tile
        ;
        uint64_t v199 = (uint64_t) v25;
        TASSIGN(v198, v199);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TCOLEXPANDMUL(v198, v196, v188);
        // pto: %q_head_dq_inline3190__tile
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v200 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %q_head_dq_inline3190__tile
        ;
        uint64_t v201 = (uint64_t) v21;
        TASSIGN(v200, v201);
        pipe_barrier(PIPE_V);
        TMUL(v200, v190, v198);
        // pto: %4
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v202 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %4
        ;
        uint64_t v203 = (uint64_t) v25;
        TASSIGN(v202, v203);
        pipe_barrier(PIPE_V);
        RoundMode v204 = RoundMode::CAST_RINT;
        SaturationMode v205 = SaturationMode::OFF;
        TCVT(v202, v200, v204, v205);
        // pto: %q_head_dq_v1_inline3220__tile
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v206 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %q_head_dq_v1_inline3220__tile
        ;
        uint64_t v207 = (uint64_t) v21;
        TASSIGN(v206, v207);
        pipe_barrier(PIPE_V);
        RoundMode v208 = RoundMode::CAST_ROUND;
        SaturationMode v209 = SaturationMode::OFF;
        TCVT(v206, v202, v208, v209);
        // pto: %q_head_sq_inline3191__tile
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v210 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %q_head_sq_inline3191__tile
        ;
        uint64_t v211 = (uint64_t) v25;
        TASSIGN(v210, v211);
        pipe_barrier(PIPE_V);
        TMUL(v210, v206, v206);
        // pto: %tmp_tile
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v212 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %tmp_tile
        ;
        uint64_t v213 = (uint64_t) v22;
        TASSIGN(v212, v213);
        // pto: %q_head_sq_row_inline3188__tile
        ;
        Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v214 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v34);
        // pto: %q_head_sq_row_inline3188__tile
        ;
        uint64_t v215 = (uint64_t) v26;
        TASSIGN(v214, v215);
        pipe_barrier(PIPE_V);
        TROWSUM(v214, v210, v212);
        // pto: %q_head_sq_sum_inline3207__tile
        ;
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v216 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v38);
        // pto: %q_head_sq_sum_inline3207__tile
        ;
        uint64_t v217 = (uint64_t) v26;
        TASSIGN(v216, v217);
        // pto: %q_head_sq_mean_inline3168__tile
        ;
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v218 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v38);
        // pto: %q_head_sq_mean_inline3168__tile
        ;
        uint64_t v219 = (uint64_t) v25;
        TASSIGN(v218, v219);
        pipe_barrier(PIPE_V);
        TMULS(v218, v216, v45);
        // pto: %q_head_var_inline3169__tile
        ;
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v220 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v38);
        // pto: %q_head_var_inline3169__tile
        ;
        uint64_t v221 = (uint64_t) v25;
        TASSIGN(v220, v221);
        pipe_barrier(PIPE_V);
        TADDS(v220, v218, v46);
        // pto: %rsqrt_tmp
        ;
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v222 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v38);
        // pto: %rsqrt_tmp
        ;
        uint64_t v223 = (uint64_t) v22;
        TASSIGN(v222, v223);
        // pto: %q_head_inv_rms_inline3185__tile
        ;
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v224 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v38);
        // pto: %q_head_inv_rms_inline3185__tile
        ;
        uint64_t v225 = (uint64_t) v26;
        TASSIGN(v224, v225);
        pipe_barrier(PIPE_V);
        TRSQRT(v224, v220, v222);
        // pto: %q_head_inv_rms_t_inline3199__tile
        ;
        Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v226 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v34);
        // pto: %q_head_inv_rms_t_inline3199__tile
        ;
        uint64_t v227 = (uint64_t) v26;
        TASSIGN(v226, v227);
        // pto: %5
        ;
        Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v228 = Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v47);
        // pto: %5
        ;
        uint64_t v229 = (uint64_t) v21;
        TASSIGN(v228, v229);
        // pto: %slice_view
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, 8, 448, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v230;
        // pto: %slice_view
        ;
        uint64_t v231 = (uint64_t) v21;
        TASSIGN(v230, v231);
        // pto: %q_nope_normed_inline3170__tile
        ;
        Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v232 = Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v47);
        // pto: %q_nope_normed_inline3170__tile
        ;
        uint64_t v233 = (uint64_t) v25;
        TASSIGN(v232, v233);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v232, v230, v226);
        // pto: %q_nope_bf16_inline3210__tile
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v234 = Tile<TileType::Vec, bfloat16_t, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v47);
        // pto: %q_nope_bf16_inline3210__tile
        ;
        uint64_t v235 = (uint64_t) v25;
        TASSIGN(v234, v235);
        pipe_barrier(PIPE_V);
        RoundMode v236 = RoundMode::CAST_RINT;
        SaturationMode v237 = SaturationMode::OFF;
        TCVT(v234, v232, v236, v237);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %q_rope_chunk_raw_inline3171__tile
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v238 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %q_rope_chunk_raw_inline3171__tile
        ;
        uint64_t v239 = (uint64_t) v27;
        TASSIGN(v238, v239);
        // pto: %69
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v240;
        // pto: %69
        ;
        uint64_t v241 = (uint64_t) v27;
        TASSIGN(v240, v241);
        // pto: %q_rope_chunk_inline3230__tile
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v242 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %q_rope_chunk_inline3230__tile
        ;
        uint64_t v243 = (uint64_t) v21;
        TASSIGN(v242, v243);
        TROWEXPANDMUL(v242, v240, v226);
        // pto: %6
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v244 = Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %6
        ;
        uint64_t v245 = (uint64_t) v22;
        TASSIGN(v244, v245);
        pipe_barrier(PIPE_V);
        RoundMode v246 = RoundMode::CAST_RINT;
        SaturationMode v247 = SaturationMode::OFF;
        TCVT(v244, v242, v246, v247);
        // pto: %q_rope_chunk_v1_inline3172__tile
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v248 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %q_rope_chunk_v1_inline3172__tile
        ;
        uint64_t v249 = (uint64_t) v21;
        TASSIGN(v248, v249);
        pipe_barrier(PIPE_V);
        RoundMode v250 = RoundMode::CAST_ROUND;
        SaturationMode v251 = SaturationMode::OFF;
        TCVT(v248, v244, v250, v251);
        // pto: %gather_acc_init
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v252 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %gather_acc_init
        ;
        uint64_t v253 = (uint64_t) v22;
        TASSIGN(v252, v253);
        for (int64_t v254 = v42; v254 < v38; v254 += v34) {
          // pto: %gather_inp_row
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v255 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v36);
          // pto: %gather_inp_row
          ;
          uint64_t v256 = (uint64_t) v21;
          TASSIGN(v255, v256);
          // pto: %70
          ;
          int64_t v257 = (int64_t) ((uint64_t) v254 * (uint64_t) v16);
          // pto: %70
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v258;
          // pto: %70
          ;
          uint64_t v259 = (uint64_t) ((int64_t) (uint64_t) v257 + (uint64_t) v21);
          TASSIGN(v258, v259);
          // pto: %gather_idx_row
          ;
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v260 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v36);
          // pto: %gather_idx_row
          ;
          uint64_t v261 = (uint64_t) v20;
          TASSIGN(v260, v261);
          // pto: %71
          ;
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v262;
          // pto: %71
          ;
          uint64_t v263 = (uint64_t) v257;
          TASSIGN(v262, v263);
          // pto: %gather_row_tmp
          ;
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v264 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v36);
          // pto: %gather_row_tmp
          ;
          uint64_t v265 = (uint64_t) v26;
          TASSIGN(v264, v265);
          // pto: %gather_row
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v266 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v36);
          // pto: %gather_row
          ;
          uint64_t v267 = (uint64_t) v28;
          TASSIGN(v266, v267);
          pipe_barrier(PIPE_V);
          TGATHER(v266, v258, v262, v264);
          // pto: %assemble_view
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v268;
          // pto: %assemble_view
          ;
          uint64_t v269 = (uint64_t) ((int64_t) (uint64_t) v257 + (uint64_t) v22);
          TASSIGN(v268, v269);
          pipe_barrier(PIPE_V);
          TMOV(v268, v266);
        };
        // pto: %q_rope_swapped_inline3193__tile
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v270 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %q_rope_swapped_inline3193__tile
        ;
        uint64_t v271 = (uint64_t) v22;
        TASSIGN(v270, v271);
        // pto: %q_rope_base_inline3211__tile
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v272 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %q_rope_base_inline3211__tile
        ;
        uint64_t v273 = (uint64_t) v21;
        TASSIGN(v272, v273);
        pipe_barrier(PIPE_V);
        TMUL(v272, v248, v128);
        // pto: %q_rope_delta_inline3215__tile
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v274 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %q_rope_delta_inline3215__tile
        ;
        uint64_t v275 = (uint64_t) v22;
        TASSIGN(v274, v275);
        TMUL(v274, v270, v137);
        // pto: %q_rope_rot_inline3182__tile
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v276 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %q_rope_rot_inline3182__tile
        ;
        uint64_t v277 = (uint64_t) v21;
        TASSIGN(v276, v277);
        pipe_barrier(PIPE_V);
        TADD(v276, v272, v274);
        // pto: %q_rope_bf16_inline3212__tile
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v278 = Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %q_rope_bf16_inline3212__tile
        ;
        uint64_t v279 = (uint64_t) v21;
        TASSIGN(v278, v279);
        pipe_barrier(PIPE_V);
        RoundMode v280 = RoundMode::CAST_RINT;
        SaturationMode v281 = SaturationMode::OFF;
        TCVT(v278, v276, v280, v281);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        // pto: %q_flat_inline3189__iter_v3_pview
        ;
        __gm__ bfloat16_t* v282 = PTOAS__GLOBAL_TENSOR_DATA(v61);
        // pto: %q_flat_inline3189__iter_v3_pview
        ;
        const int64_t v283 = 0;
        // pto: %q_flat_inline3189__iter_v3_pview
        ;
        const int64_t v284 = 32768;
        // pto: %q_flat_inline3189__iter_v3_pview
        ;
        pto::Shape<1, 1, 1, 8, 448> v285 = pto::Shape<1, 1, 1, 8, 448>();
        // pto: %q_flat_inline3189__iter_v3_pview
        ;
        pto::Stride<262144, 262144, 262144, 32768, 1> v286 = pto::Stride<262144, 262144, 262144, 32768, 1>();
        // pto: %q_flat_inline3189__iter_v3_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 448>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND> v287 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 448>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND>(v282 + (v283 + v130 * v284 + v158), v285, v286);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v287, v234);
        // pto: %76
        ;
        int64_t v288 = (int64_t) ((uint64_t) v154 + (uint64_t) v47);
        // pto: %77
        ;
        int64_t v289 = v288 < v42 ? v42 : v288;
        // pto: %q_flat_inline3189__tile_pview
        ;
        __gm__ bfloat16_t* v290 = PTOAS__GLOBAL_TENSOR_DATA(v61);
        // pto: %q_flat_inline3189__tile_pview
        ;
        const int64_t v291 = 0;
        // pto: %q_flat_inline3189__tile_pview
        ;
        const int64_t v292 = 32768;
        // pto: %q_flat_inline3189__tile_pview
        ;
        pto::Shape<1, 1, 1, 8, 64> v293 = pto::Shape<1, 1, 1, 8, 64>();
        // pto: %q_flat_inline3189__tile_pview
        ;
        pto::Stride<262144, 262144, 262144, 32768, 1> v294 = pto::Stride<262144, 262144, 262144, 32768, 1>();
        // pto: %q_flat_inline3189__tile_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND> v295 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND>(v290 + (v291 + v130 * v292 + v289), v293, v294);
        pipe_barrier(PIPE_MTE3);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        TSTORE(v295, v278);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        // pto: %7
        ;
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v296 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
        // pto: %7
        ;
        uint64_t v297 = (uint64_t) v24;
        TASSIGN(v296, v297);
        // pto: %8
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v298 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %8
        ;
        uint64_t v299 = (uint64_t) v23;
        TASSIGN(v298, v299);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        RoundMode v300 = RoundMode::CAST_NONE;
        SaturationMode v301 = SaturationMode::OFF;
        TCVT(v298, v172, v300, v301);
        // pto: %9
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v302 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %9
        ;
        uint64_t v303 = (uint64_t) v29;
        TASSIGN(v302, v303);
        TEXPANDS(v302, v44);
        // pto: %10
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v304 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %10
        ;
        uint64_t v305 = (uint64_t) v29;
        TASSIGN(v304, v305);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v304, v302, v120);
        // pto: %11
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v306 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %11
        ;
        uint64_t v307 = (uint64_t) v29;
        TASSIGN(v306, v307);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        TCOLEXPANDMUL(v306, v304, v296);
        // pto: %12
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v308 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %12
        ;
        uint64_t v309 = (uint64_t) v23;
        TASSIGN(v308, v309);
        pipe_barrier(PIPE_V);
        TMUL(v308, v298, v306);
        // pto: %13
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v310 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %13
        ;
        uint64_t v311 = (uint64_t) v29;
        TASSIGN(v310, v311);
        pipe_barrier(PIPE_V);
        RoundMode v312 = RoundMode::CAST_RINT;
        SaturationMode v313 = SaturationMode::OFF;
        TCVT(v310, v308, v312, v313);
        // pto: %14
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v314 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %14
        ;
        uint64_t v315 = (uint64_t) v23;
        TASSIGN(v314, v315);
        pipe_barrier(PIPE_V);
        RoundMode v316 = RoundMode::CAST_ROUND;
        SaturationMode v317 = SaturationMode::OFF;
        TCVT(v314, v310, v316, v317);
        // pto: %15
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v318 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %15
        ;
        uint64_t v319 = (uint64_t) v29;
        TASSIGN(v318, v319);
        pipe_barrier(PIPE_V);
        TMUL(v318, v314, v314);
        // pto: %16
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v320 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %16
        ;
        uint64_t v321 = (uint64_t) v24;
        TASSIGN(v320, v321);
        // pto: %17
        ;
        Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v322 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v34);
        // pto: %17
        ;
        uint64_t v323 = (uint64_t) v30;
        TASSIGN(v322, v323);
        pipe_barrier(PIPE_V);
        TROWSUM(v322, v318, v320);
        // pto: %18
        ;
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v324 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v38);
        // pto: %18
        ;
        uint64_t v325 = (uint64_t) v30;
        TASSIGN(v324, v325);
        // pto: %19
        ;
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v326 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v38);
        // pto: %19
        ;
        uint64_t v327 = (uint64_t) v29;
        TASSIGN(v326, v327);
        pipe_barrier(PIPE_V);
        TMULS(v326, v324, v45);
        // pto: %20
        ;
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v328 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v38);
        // pto: %20
        ;
        uint64_t v329 = (uint64_t) v29;
        TASSIGN(v328, v329);
        pipe_barrier(PIPE_V);
        TADDS(v328, v326, v46);
        // pto: %21
        ;
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v330 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v38);
        // pto: %21
        ;
        uint64_t v331 = (uint64_t) v24;
        TASSIGN(v330, v331);
        // pto: %22
        ;
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v332 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v38);
        // pto: %22
        ;
        uint64_t v333 = (uint64_t) v30;
        TASSIGN(v332, v333);
        pipe_barrier(PIPE_V);
        TRSQRT(v332, v328, v330);
        // pto: %23
        ;
        Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v334 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v34);
        // pto: %23
        ;
        uint64_t v335 = (uint64_t) v30;
        TASSIGN(v334, v335);
        // pto: %24
        ;
        Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v336 = Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v47);
        // pto: %24
        ;
        uint64_t v337 = (uint64_t) v23;
        TASSIGN(v336, v337);
        // pto: %78
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, 8, 448, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v338;
        // pto: %78
        ;
        uint64_t v339 = (uint64_t) v23;
        TASSIGN(v338, v339);
        // pto: %25
        ;
        Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v340 = Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v47);
        // pto: %25
        ;
        uint64_t v341 = (uint64_t) v29;
        TASSIGN(v340, v341);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v340, v338, v334);
        // pto: %26
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v342 = Tile<TileType::Vec, bfloat16_t, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v47);
        // pto: %26
        ;
        uint64_t v343 = (uint64_t) v29;
        TASSIGN(v342, v343);
        pipe_barrier(PIPE_V);
        RoundMode v344 = RoundMode::CAST_RINT;
        SaturationMode v345 = SaturationMode::OFF;
        TCVT(v342, v340, v344, v345);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        // pto: %27
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v346 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %27
        ;
        uint64_t v347 = (uint64_t) v31;
        TASSIGN(v346, v347);
        // pto: %79
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v348;
        // pto: %79
        ;
        uint64_t v349 = (uint64_t) v31;
        TASSIGN(v348, v349);
        // pto: %28
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v350 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %28
        ;
        uint64_t v351 = (uint64_t) v23;
        TASSIGN(v350, v351);
        TROWEXPANDMUL(v350, v348, v334);
        // pto: %29
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v352 = Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %29
        ;
        uint64_t v353 = (uint64_t) v24;
        TASSIGN(v352, v353);
        pipe_barrier(PIPE_V);
        RoundMode v354 = RoundMode::CAST_RINT;
        SaturationMode v355 = SaturationMode::OFF;
        TCVT(v352, v350, v354, v355);
        // pto: %30
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v356 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %30
        ;
        uint64_t v357 = (uint64_t) v23;
        TASSIGN(v356, v357);
        pipe_barrier(PIPE_V);
        RoundMode v358 = RoundMode::CAST_ROUND;
        SaturationMode v359 = SaturationMode::OFF;
        TCVT(v356, v352, v358, v359);
        // pto: %31
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v360 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %31
        ;
        uint64_t v361 = (uint64_t) v24;
        TASSIGN(v360, v361);
        for (int64_t v362 = v42; v362 < v38; v362 += v34) {
          // pto: %32
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v363 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v36);
          // pto: %32
          ;
          uint64_t v364 = (uint64_t) v23;
          TASSIGN(v363, v364);
          // pto: %81
          ;
          int64_t v365 = (int64_t) ((uint64_t) v362 * (uint64_t) v16);
          // pto: %81
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v366;
          // pto: %81
          ;
          uint64_t v367 = (uint64_t) ((int64_t) (uint64_t) v365 + (uint64_t) v23);
          TASSIGN(v366, v367);
          // pto: %33
          ;
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v368 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v36);
          // pto: %33
          ;
          uint64_t v369 = (uint64_t) v20;
          TASSIGN(v368, v369);
          // pto: %82
          ;
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v370;
          // pto: %82
          ;
          uint64_t v371 = (uint64_t) v365;
          TASSIGN(v370, v371);
          // pto: %34
          ;
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v372 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v36);
          // pto: %34
          ;
          uint64_t v373 = (uint64_t) v30;
          TASSIGN(v372, v373);
          // pto: %35
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v374 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v36);
          // pto: %35
          ;
          uint64_t v375 = (uint64_t) v32;
          TASSIGN(v374, v375);
          pipe_barrier(PIPE_V);
          TGATHER(v374, v366, v370, v372);
          // pto: %83
          ;
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v376;
          // pto: %83
          ;
          uint64_t v377 = (uint64_t) ((int64_t) (uint64_t) v365 + (uint64_t) v24);
          TASSIGN(v376, v377);
          pipe_barrier(PIPE_V);
          TMOV(v376, v374);
        };
        // pto: %37
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v378 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %37
        ;
        uint64_t v379 = (uint64_t) v24;
        TASSIGN(v378, v379);
        // pto: %38
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v380 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %38
        ;
        uint64_t v381 = (uint64_t) v23;
        TASSIGN(v380, v381);
        pipe_barrier(PIPE_V);
        TMUL(v380, v356, v128);
        // pto: %39
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v382 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %39
        ;
        uint64_t v383 = (uint64_t) v24;
        TASSIGN(v382, v383);
        TMUL(v382, v378, v137);
        // pto: %40
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v384 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %40
        ;
        uint64_t v385 = (uint64_t) v23;
        TASSIGN(v384, v385);
        pipe_barrier(PIPE_V);
        TADD(v384, v380, v382);
        // pto: %41
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v386 = Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %41
        ;
        uint64_t v387 = (uint64_t) v23;
        TASSIGN(v386, v387);
        pipe_barrier(PIPE_V);
        RoundMode v388 = RoundMode::CAST_RINT;
        SaturationMode v389 = SaturationMode::OFF;
        TCVT(v386, v384, v388, v389);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        // pto: %87
        ;
        __gm__ bfloat16_t* v390 = PTOAS__GLOBAL_TENSOR_DATA(v61);
        // pto: %87
        ;
        const int64_t v391 = 0;
        // pto: %87
        ;
        const int64_t v392 = 32768;
        // pto: %87
        ;
        pto::Shape<1, 1, 1, 8, 448> v393 = pto::Shape<1, 1, 1, 8, 448>();
        // pto: %87
        ;
        pto::Stride<262144, 262144, 262144, 32768, 1> v394 = pto::Stride<262144, 262144, 262144, 32768, 1>();
        // pto: %87
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 448>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND> v395 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 448>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND>(v390 + (v391 + v130 * v392 + v174), v393, v394);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v395, v342);
        // pto: %90
        ;
        int64_t v396 = (int64_t) ((uint64_t) v155 + (uint64_t) v47);
        // pto: %91
        ;
        int64_t v397 = v396 < v42 ? v42 : v396;
        // pto: %92
        ;
        __gm__ bfloat16_t* v398 = PTOAS__GLOBAL_TENSOR_DATA(v61);
        // pto: %92
        ;
        const int64_t v399 = 0;
        // pto: %92
        ;
        const int64_t v400 = 32768;
        // pto: %92
        ;
        pto::Shape<1, 1, 1, 8, 64> v401 = pto::Shape<1, 1, 1, 8, 64>();
        // pto: %92
        ;
        pto::Stride<262144, 262144, 262144, 32768, 1> v402 = pto::Stride<262144, 262144, 262144, 32768, 1>();
        // pto: %92
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND> v403 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND>(v398 + (v399 + v130 * v400 + v397), v401, v402);
        pipe_barrier(PIPE_MTE3);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        TSTORE(v403, v386);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      };
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
    } else {
      // pto: %93
      ;
      int64_t v404 = (int64_t) ((uint64_t) v8 - (uint64_t) v118);
      // pto: %qr_scale_dq_tail_inline3214__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v405 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v34);
      // pto: %qr_scale_dq_tail_inline3214__ssa_v0
      ;
      uint64_t v406 = (uint64_t) v20;
      TASSIGN(v405, v406);
      // pto: %94
      ;
      int64_t v407 = v118 < v42 ? v42 : v118;
      // pto: %95
      ;
      __gm__ float* v408 = PTOAS__GLOBAL_TENSOR_DATA(v69);
      // pto: %95
      ;
      const int64_t v409 = 0;
      // pto: %95
      ;
      pto::Shape<1, 1, 1, 8, 1> v410 = pto::Shape<1, 1, 1, 8, 1>();
      // pto: %95
      ;
      pto::Stride<8, 8, 8, 1, 512> v411 = pto::Stride<8, 8, 8, 1, 512>();
      // pto: %95
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<8, 8, 8, 1, 512>, pto::Layout::DN> v412 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<8, 8, 8, 1, 512>, pto::Layout::DN>(v408 + (v409 + v407), v410, v411);
      pipe_barrier(PIPE_ALL);
      TLOAD(v405, v412);
      // pto: %q_cos_il_tail_inline3217__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v413 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v404, v36);
      // pto: %q_cos_il_tail_inline3217__ssa_v0
      ;
      uint64_t v414 = (uint64_t) v22;
      TASSIGN(v413, v414);
      // pto: %96
      ;
      int64_t v415 = v119 < v42 ? v42 : v119;
      // pto: %97
      ;
      const int64_t v416 = 0;
      // pto: %97
      ;
      __gm__ float* v417 = PTOAS__GLOBAL_TENSOR_DATA(v78);
      // pto: %97
      ;
      const int64_t v418 = 1;
      // pto: %97
      ;
      const int64_t v419 = 1;
      // pto: %97
      ;
      const int64_t v420 = 1;
      // pto: %97
      ;
      int64_t v421 = v404 * v36;
      // pto: %97
      ;
      int64_t v422 = v420 * v421;
      // pto: %97
      ;
      pto::Shape<1, 1, 1, -1, 64> v423 = pto::Shape<1, 1, 1, -1, 64>(v418, v419, v420, v404, v36);
      // pto: %97
      ;
      pto::Stride<-1, -1, -1, -1, -1> v424 = pto::Stride<-1, -1, -1, -1, -1>(v419 * v422, v422, v421, v36, v34);
      // pto: %97
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v425 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v417 + (v416 + v415 * v36 + v42 * v34), v423, v424);
      pipe_barrier(PIPE_ALL);
      TLOAD(v413, v425);
      // pto: %q_sin_signed_tail_inline3221__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v426 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v404, v36);
      // pto: %q_sin_signed_tail_inline3221__ssa_v0
      ;
      uint64_t v427 = (uint64_t) v29;
      TASSIGN(v426, v427);
      // pto: %99
      ;
      const int64_t v428 = 0;
      // pto: %99
      ;
      __gm__ float* v429 = PTOAS__GLOBAL_TENSOR_DATA(v87);
      // pto: %99
      ;
      const int64_t v430 = 1;
      // pto: %99
      ;
      const int64_t v431 = 1;
      // pto: %99
      ;
      const int64_t v432 = 1;
      // pto: %99
      ;
      int64_t v433 = v404 * v36;
      // pto: %99
      ;
      int64_t v434 = v432 * v433;
      // pto: %99
      ;
      pto::Shape<1, 1, 1, -1, 64> v435 = pto::Shape<1, 1, 1, -1, 64>(v430, v431, v432, v404, v36);
      // pto: %99
      ;
      pto::Stride<-1, -1, -1, -1, -1> v436 = pto::Stride<-1, -1, -1, -1, -1>(v431 * v434, v434, v433, v36, v34);
      // pto: %99
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v437 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v429 + (v428 + v415 * v36 + v42 * v34), v435, v436);
      pipe_barrier(PIPE_ALL);
      TLOAD(v426, v437);
      // pto: %t__tmp_v53
      ;
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v438 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
      // pto: %t__tmp_v53
      ;
      uint64_t v439 = (uint64_t) v21;
      TASSIGN(v438, v439);
      pipe_barrier(PIPE_V);
      TEXPANDS(v438, v44);
      // pto: %t__ci_tmp_v0
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v440 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v48);
      // pto: %t__ci_tmp_v0
      ;
      uint64_t v441 = (uint64_t) v23;
      TASSIGN(v440, v441);
      // pto: %t__tmp_v54
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v442 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v36);
      // pto: %t__tmp_v54
      ;
      uint64_t v443 = (uint64_t) v25;
      TASSIGN(v442, v443);
      // pto: %ci_tmp_view
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v444;
      TRESHAPE(v444, v440);
      // pto: %ci_dst_view
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v445;
      TRESHAPE(v445, v442);
      wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      TCI<Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v445, v49, v444);
      set_flag(PIPE_S, PIPE_V, EVENT_ID0);
      // pto: %t__tmp_v55
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v446 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v36);
      // pto: %t__tmp_v55
      ;
      uint64_t v447 = (uint64_t) v23;
      TASSIGN(v446, v447);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
      RoundMode v448 = RoundMode::CAST_ROUND;
      SaturationMode v449 = SaturationMode::OFF;
      TCVT(v446, v442, v448, v449);
      // pto: %q_col_inline3202__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v450 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
      // pto: %q_col_inline3202__ssa_v0
      ;
      uint64_t v451 = (uint64_t) v21;
      TASSIGN(v450, v451);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v450, v438, v446);
      // pto: %t__tmp_v56
      ;
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v452 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
      // pto: %t__tmp_v56
      ;
      uint64_t v453 = (uint64_t) v23;
      TASSIGN(v452, v453);
      pipe_barrier(PIPE_V);
      TMULS(v452, v450, v50);
      // pto: %t__tmp_v57
      ;
      Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v454 = Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
      // pto: %t__tmp_v57
      ;
      uint64_t v455 = (uint64_t) v23;
      TASSIGN(v454, v455);
      pipe_barrier(PIPE_V);
      RoundMode v456 = RoundMode::CAST_TRUNC;
      SaturationMode v457 = SaturationMode::ON;
      TCVT(v454, v452, v456, v457);
      // pto: %q_dup_f_inline3187__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v458 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
      // pto: %q_dup_f_inline3187__ssa_v0
      ;
      uint64_t v459 = (uint64_t) v23;
      TASSIGN(v458, v459);
      pipe_barrier(PIPE_V);
      RoundMode v460 = RoundMode::CAST_ROUND;
      SaturationMode v461 = SaturationMode::OFF;
      TCVT(v458, v454, v460, v461);
      // pto: %t__tmp_v58
      ;
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v462 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
      // pto: %t__tmp_v58
      ;
      uint64_t v463 = (uint64_t) v23;
      TASSIGN(v462, v463);
      pipe_barrier(PIPE_V);
      TMULS(v462, v458, v51);
      // pto: %q_lane_inline3224__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v464 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
      // pto: %q_lane_inline3224__ssa_v0
      ;
      uint64_t v465 = (uint64_t) v23;
      TASSIGN(v464, v465);
      pipe_barrier(PIPE_V);
      TSUB(v464, v450, v462);
      // pto: %t__tmp_v59
      ;
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v466 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
      // pto: %t__tmp_v59
      ;
      uint64_t v467 = (uint64_t) v21;
      TASSIGN(v466, v467);
      pipe_barrier(PIPE_V);
      TADDS(v466, v450, v44);
      // pto: %t__tmp_v60
      ;
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v468 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
      // pto: %t__tmp_v60
      ;
      uint64_t v469 = (uint64_t) v23;
      TASSIGN(v468, v469);
      TMULS(v468, v464, v51);
      // pto: %q_swap_f_inline3183__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v470 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
      // pto: %q_swap_f_inline3183__ssa_v0
      ;
      uint64_t v471 = (uint64_t) v21;
      TASSIGN(v470, v471);
      pipe_barrier(PIPE_V);
      TSUB(v470, v466, v468);
      set_flag(PIPE_V, PIPE_S, EVENT_ID1);
      // pto: %t__ci_tmp_v1
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v472 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v48);
      // pto: %t__ci_tmp_v1
      ;
      uint64_t v473 = (uint64_t) v23;
      TASSIGN(v472, v473);
      // pto: %t__tmp_v61
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v474 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v38);
      // pto: %t__tmp_v61
      ;
      uint64_t v475 = (uint64_t) v25;
      TASSIGN(v474, v475);
      // pto: %100
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v476;
      TRESHAPE(v476, v472);
      // pto: %101
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v477;
      TRESHAPE(v477, v474);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
      TCI<Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v477, v49, v476);
      set_flag(PIPE_S, PIPE_V, EVENT_ID1);
      // pto: %t__tmp_v62
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v478 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v38);
      // pto: %t__tmp_v62
      ;
      uint64_t v479 = (uint64_t) v23;
      TASSIGN(v478, v479);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
      RoundMode v480 = RoundMode::CAST_ROUND;
      SaturationMode v481 = SaturationMode::OFF;
      TCVT(v478, v474, v480, v481);
      // pto: %q_row_seed_inline3195__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v482 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v38);
      // pto: %q_row_seed_inline3195__ssa_v0
      ;
      uint64_t v483 = (uint64_t) v25;
      TASSIGN(v482, v483);
      pipe_barrier(PIPE_V);
      TMULS(v482, v478, v52);
      // pto: %t__tmp_v63
      ;
      Tile<TileType::Vec, float, 64, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v484 = Tile<TileType::Vec, float, 64, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v38);
      // pto: %t__tmp_v63
      ;
      uint64_t v485 = (uint64_t) v23;
      TASSIGN(v484, v485);
      pipe_barrier(PIPE_V);
      TEXPANDS(v484, v44);
      // pto: %q_row_grid_inline3227__ssa_v0
      ;
      Tile<TileType::Vec, float, 64, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v486 = Tile<TileType::Vec, float, 64, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v38);
      // pto: %q_row_grid_inline3227__ssa_v0
      ;
      uint64_t v487 = (uint64_t) v23;
      TASSIGN(v486, v487);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v486, v484, v482);
      // pto: %transpose_tmp
      ;
      Tile<TileType::Vec, float, 64, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v488 = Tile<TileType::Vec, float, 64, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v38);
      // pto: %transpose_tmp
      ;
      uint64_t v489 = (uint64_t) v25;
      TASSIGN(v488, v489);
      // pto: %q_row_offset_inline3225__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v490 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
      // pto: %q_row_offset_inline3225__ssa_v0
      ;
      uint64_t v491 = (uint64_t) v24;
      TASSIGN(v490, v491);
      pipe_barrier(PIPE_V);
      TTRANS(v490, v486, v488);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      // pto: %t__tmp_v64
      ;
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v492 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
      // pto: %t__tmp_v64
      ;
      uint64_t v493 = (uint64_t) v21;
      TASSIGN(v492, v493);
      pipe_barrier(PIPE_V);
      TADD(v492, v470, v490);
      // pto: %q_swap_idx_tail_inline3226__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v494 = Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
      // pto: %q_swap_idx_tail_inline3226__ssa_v0
      ;
      uint64_t v495 = (uint64_t) v24;
      TASSIGN(v494, v495);
      pipe_barrier(PIPE_V);
      RoundMode v496 = RoundMode::CAST_ROUND;
      SaturationMode v497 = SaturationMode::ON;
      TCVT(v494, v492, v496, v497);
      // pto: %q_head_reduce_tmp_inline3209__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v498 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
      // pto: %q_head_reduce_tmp_inline3209__ssa_v0
      ;
      uint64_t v499 = (uint64_t) v21;
      TASSIGN(v498, v499);
      // pto: %q_gather_tmp_inline3218__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v500 = Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
      // pto: %q_gather_tmp_inline3218__ssa_v0
      ;
      uint64_t v501 = (uint64_t) v18;
      TASSIGN(v500, v501);
      pipe_barrier(PIPE_ALL);
      for (int64_t v502 = v42; v502 < v41; v502 += v34) {
        // pto: %102, %103
        ;
        int64_t v503 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v117 + (uint64_t) v502) * (uint64_t) v35);
        // pto: %q_head_acc_tail_inline3177__ssa_v0
        ;
        Tile<TileType::Vec, int32_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v504 = Tile<TileType::Vec, int32_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %q_head_acc_tail_inline3177__ssa_v0
        ;
        uint64_t v505 = (uint64_t) v23;
        TASSIGN(v504, v505);
        // pto: %105
        ;
        int64_t v506 = v503 < v42 ? v42 : v503;
        // pto: %106
        ;
        __gm__ int32_t* v507 = PTOAS__GLOBAL_TENSOR_DATA(v105);
        // pto: %106
        ;
        const int64_t v508 = 0;
        // pto: %106
        ;
        const int64_t v509 = 32768;
        // pto: %106
        ;
        pto::Shape<1, 1, 1, 8, 512> v510 = pto::Shape<1, 1, 1, 8, 512>();
        // pto: %106
        ;
        pto::Stride<262144, 262144, 262144, 32768, 1> v511 = pto::Stride<262144, 262144, 262144, 32768, 1>();
        // pto: %106
        ;
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND> v512 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND>(v507 + (v508 + v407 * v509 + v506), v510, v511);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
        TLOAD(v504, v512);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        // pto: %q_head_scale_input_tail_inline3222__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v513 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
        // pto: %q_head_scale_input_tail_inline3222__ssa_v0
        ;
        uint64_t v514 = (uint64_t) v19;
        TASSIGN(v513, v514);
        // pto: %108
        ;
        __gm__ float* v515 = PTOAS__GLOBAL_TENSOR_DATA(v115);
        // pto: %108
        ;
        const int64_t v516 = 0;
        // pto: %108
        ;
        pto::Shape<1, 1, 1, 1, 512> v517 = pto::Shape<1, 1, 1, 1, 512>();
        // pto: %108
        ;
        pto::Stride<512, 512, 512, 512, 1> v518 = pto::Stride<512, 512, 512, 512, 1>();
        // pto: %108
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v519 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v515 + (v516 + v506), v517, v518);
        TLOAD(v513, v519);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        // pto: %q_head_scale_tail_inline3229__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v520 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v35);
        // pto: %q_head_scale_tail_inline3229__ssa_v0
        ;
        uint64_t v521 = (uint64_t) v19;
        TASSIGN(v520, v521);
        // pto: %q_head_acc_fp32_tail_inline3231__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v522 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %q_head_acc_fp32_tail_inline3231__ssa_v0
        ;
        uint64_t v523 = (uint64_t) v23;
        TASSIGN(v522, v523);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        RoundMode v524 = RoundMode::CAST_NONE;
        SaturationMode v525 = SaturationMode::OFF;
        TCVT(v522, v504, v524, v525);
        // pto: %t__tmp_v65
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v526 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %t__tmp_v65
        ;
        uint64_t v527 = (uint64_t) v25;
        TASSIGN(v526, v527);
        TEXPANDS(v526, v44);
        // pto: %t__tmp_v66
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v528 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %t__tmp_v66
        ;
        uint64_t v529 = (uint64_t) v25;
        TASSIGN(v528, v529);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v528, v526, v405);
        // pto: %q_head_scale_combined_tail_inline3232__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v530 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %q_head_scale_combined_tail_inline3232__ssa_v0
        ;
        uint64_t v531 = (uint64_t) v25;
        TASSIGN(v530, v531);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        TCOLEXPANDMUL(v530, v528, v520);
        // pto: %q_head_dq_tail_inline3166__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v532 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %q_head_dq_tail_inline3166__ssa_v0
        ;
        uint64_t v533 = (uint64_t) v23;
        TASSIGN(v532, v533);
        pipe_barrier(PIPE_V);
        TMUL(v532, v522, v530);
        // pto: %t__tmp_v67
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v534 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %t__tmp_v67
        ;
        uint64_t v535 = (uint64_t) v25;
        TASSIGN(v534, v535);
        pipe_barrier(PIPE_V);
        RoundMode v536 = RoundMode::CAST_RINT;
        SaturationMode v537 = SaturationMode::OFF;
        TCVT(v534, v532, v536, v537);
        // pto: %q_head_dq_tail_v1_inline3181__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v538 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %q_head_dq_tail_v1_inline3181__ssa_v0
        ;
        uint64_t v539 = (uint64_t) v23;
        TASSIGN(v538, v539);
        pipe_barrier(PIPE_V);
        RoundMode v540 = RoundMode::CAST_ROUND;
        SaturationMode v541 = SaturationMode::OFF;
        TCVT(v538, v534, v540, v541);
        // pto: %q_head_sq_tail_inline3165__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v542 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v35);
        // pto: %q_head_sq_tail_inline3165__ssa_v0
        ;
        uint64_t v543 = (uint64_t) v25;
        TASSIGN(v542, v543);
        pipe_barrier(PIPE_V);
        TMUL(v542, v538, v538);
        // pto: %q_head_sq_sum_tail_inline3179__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v544 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v34);
        // pto: %q_head_sq_sum_tail_inline3179__ssa_v0
        ;
        uint64_t v545 = (uint64_t) v19;
        TASSIGN(v544, v545);
        pipe_barrier(PIPE_V);
        TROWSUM(v544, v542, v498);
        // pto: %t__rm_a0_tmp_v0
        ;
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v546 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v38);
        // pto: %t__rm_a0_tmp_v0
        ;
        uint64_t v547 = (uint64_t) v19;
        TASSIGN(v546, v547);
        // pto: %t__row_major_tmp_v1
        ;
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v548 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v38);
        // pto: %t__row_major_tmp_v1
        ;
        uint64_t v549 = (uint64_t) v25;
        TASSIGN(v548, v549);
        pipe_barrier(PIPE_V);
        TMULS(v548, v546, v45);
        // pto: %t__tmp_v68
        ;
        Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v550 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v34);
        // pto: %t__tmp_v68
        ;
        uint64_t v551 = (uint64_t) v25;
        TASSIGN(v550, v551);
        // pto: %t__rm_a0_tmp_v2
        ;
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v552 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v38);
        // pto: %t__rm_a0_tmp_v2
        ;
        uint64_t v553 = (uint64_t) v25;
        TASSIGN(v552, v553);
        // pto: %t__row_major_tmp_v3
        ;
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v554 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v38);
        // pto: %t__row_major_tmp_v3
        ;
        uint64_t v555 = (uint64_t) v25;
        TASSIGN(v554, v555);
        pipe_barrier(PIPE_V);
        TADDS(v554, v552, v46);
        // pto: %t__tmp_v69
        ;
        Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v556 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v34);
        // pto: %t__tmp_v69
        ;
        uint64_t v557 = (uint64_t) v25;
        TASSIGN(v556, v557);
        // pto: %t__rm_a0_tmp_v4
        ;
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v558 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v38);
        // pto: %t__rm_a0_tmp_v4
        ;
        uint64_t v559 = (uint64_t) v25;
        TASSIGN(v558, v559);
        // pto: %t__row_major_tmp_v5
        ;
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v560 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v38);
        // pto: %t__row_major_tmp_v5
        ;
        uint64_t v561 = (uint64_t) v25;
        TASSIGN(v560, v561);
        pipe_barrier(PIPE_V);
        TSQRT(v560, v558);
        // pto: %t__tmp_v70
        ;
        Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v562 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v34);
        // pto: %t__tmp_v70
        ;
        uint64_t v563 = (uint64_t) v25;
        TASSIGN(v562, v563);
        // pto: %q_head_inv_rms_tail_inline3174__rm_a0_tmp_v6
        ;
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v564 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v38);
        // pto: %q_head_inv_rms_tail_inline3174__rm_a0_tmp_v6
        ;
        uint64_t v565 = (uint64_t) v25;
        TASSIGN(v564, v565);
        // pto: %q_head_inv_rms_tail_inline3174__row_major_tmp_v7
        ;
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v566 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v38);
        // pto: %q_head_inv_rms_tail_inline3174__row_major_tmp_v7
        ;
        uint64_t v567 = (uint64_t) v19;
        TASSIGN(v566, v567);
        pipe_barrier(PIPE_V);
        TRECIP(v566, v564);
        // pto: %q_head_inv_rms_tail_inline3174__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v568 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v34);
        // pto: %q_head_inv_rms_tail_inline3174__ssa_v0
        ;
        uint64_t v569 = (uint64_t) v19;
        TASSIGN(v568, v569);
        // pto: %t__tmp_v71
        ;
        Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v570 = Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v47);
        // pto: %t__tmp_v71
        ;
        uint64_t v571 = (uint64_t) v23;
        TASSIGN(v570, v571);
        // pto: %109
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, 8, 448, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v572;
        // pto: %109
        ;
        uint64_t v573 = (uint64_t) v23;
        TASSIGN(v572, v573);
        // pto: %q_nope_normed_tail_inline3167__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v574 = Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v47);
        // pto: %q_nope_normed_tail_inline3167__ssa_v0
        ;
        uint64_t v575 = (uint64_t) v25;
        TASSIGN(v574, v575);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v574, v572, v568);
        // pto: %q_nope_bf16_tail_inline3163__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v576 = Tile<TileType::Vec, bfloat16_t, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v47);
        // pto: %q_nope_bf16_tail_inline3163__ssa_v0
        ;
        uint64_t v577 = (uint64_t) v25;
        TASSIGN(v576, v577);
        pipe_barrier(PIPE_V);
        RoundMode v578 = RoundMode::CAST_RINT;
        SaturationMode v579 = SaturationMode::OFF;
        TCVT(v576, v574, v578, v579);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
        v576.SetValidShape(v404, v47);
        // pto: %q_flat_inline3189__iter_v1_pview
        ;
        const int64_t v580 = 0;
        // pto: %q_flat_inline3189__iter_v1_pview
        ;
        __gm__ bfloat16_t* v581 = PTOAS__GLOBAL_TENSOR_DATA(v61);
        // pto: %q_flat_inline3189__iter_v1_pview
        ;
        const int64_t v582 = 1;
        // pto: %q_flat_inline3189__iter_v1_pview
        ;
        const int64_t v583 = 1;
        // pto: %q_flat_inline3189__iter_v1_pview
        ;
        const int64_t v584 = 1;
        // pto: %q_flat_inline3189__iter_v1_pview
        ;
        int64_t v585 = v404 * v33;
        // pto: %q_flat_inline3189__iter_v1_pview
        ;
        int64_t v586 = v584 * v585;
        // pto: %q_flat_inline3189__iter_v1_pview
        ;
        pto::Shape<1, 1, 1, -1, 448> v587 = pto::Shape<1, 1, 1, -1, 448>(v582, v583, v584, v404, v47);
        // pto: %q_flat_inline3189__iter_v1_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v588 = pto::Stride<-1, -1, -1, -1, -1>(v583 * v586, v586, v585, v33, v34);
        // pto: %q_flat_inline3189__iter_v1_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 448>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v589 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 448>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v581 + (v580 + v415 * v33 + v506 * v34), v587, v588);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
        TSTORE(v589, v576);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        // pto: %q_rope_chunk_raw_tail_inline3161__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v590 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %q_rope_chunk_raw_tail_inline3161__ssa_v0
        ;
        uint64_t v591 = (uint64_t) v31;
        TASSIGN(v590, v591);
        // pto: %112
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v592;
        // pto: %112
        ;
        uint64_t v593 = (uint64_t) v31;
        TASSIGN(v592, v593);
        // pto: %q_rope_chunk_tail_inline3160__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v594 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %q_rope_chunk_tail_inline3160__ssa_v0
        ;
        uint64_t v595 = (uint64_t) v23;
        TASSIGN(v594, v595);
        TROWEXPANDMUL(v594, v592, v568);
        // pto: %t__tmp_v72
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v596 = Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %t__tmp_v72
        ;
        uint64_t v597 = (uint64_t) v25;
        TASSIGN(v596, v597);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        RoundMode v598 = RoundMode::CAST_RINT;
        SaturationMode v599 = SaturationMode::OFF;
        TCVT(v596, v594, v598, v599);
        // pto: %q_rope_chunk_tail_v1_inline3178__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v600 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %q_rope_chunk_tail_v1_inline3178__ssa_v0
        ;
        uint64_t v601 = (uint64_t) v23;
        TASSIGN(v600, v601);
        pipe_barrier(PIPE_V);
        RoundMode v602 = RoundMode::CAST_ROUND;
        SaturationMode v603 = SaturationMode::OFF;
        TCVT(v600, v596, v602, v603);
        // pto: %q_rope_swapped_tail_inline3186__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v604 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %q_rope_swapped_tail_inline3186__ssa_v0
        ;
        uint64_t v605 = (uint64_t) v25;
        TASSIGN(v604, v605);
        pipe_barrier(PIPE_V);
        TGATHER(v604, v600, v494, v500);
        // pto: %q_rope_base_tail_inline3159__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v606 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %q_rope_base_tail_inline3159__ssa_v0
        ;
        uint64_t v607 = (uint64_t) v23;
        TASSIGN(v606, v607);
        pipe_barrier(PIPE_V);
        TMUL(v606, v600, v413);
        // pto: %q_rope_delta_tail_inline3219__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v608 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %q_rope_delta_tail_inline3219__ssa_v0
        ;
        uint64_t v609 = (uint64_t) v25;
        TASSIGN(v608, v609);
        TMUL(v608, v604, v426);
        // pto: %q_rope_rot_tail_inline3158__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v610 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %q_rope_rot_tail_inline3158__ssa_v0
        ;
        uint64_t v611 = (uint64_t) v23;
        TASSIGN(v610, v611);
        pipe_barrier(PIPE_V);
        TADD(v610, v606, v608);
        // pto: %q_rope_bf16_tail_inline3164__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v612 = Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v38, v36);
        // pto: %q_rope_bf16_tail_inline3164__ssa_v0
        ;
        uint64_t v613 = (uint64_t) v23;
        TASSIGN(v612, v613);
        pipe_barrier(PIPE_V);
        RoundMode v614 = RoundMode::CAST_RINT;
        SaturationMode v615 = SaturationMode::OFF;
        TCVT(v612, v610, v614, v615);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
        v612.SetValidShape(v404, v36);
        // pto: %114
        ;
        int64_t v616 = (int64_t) ((uint64_t) v503 + (uint64_t) v47);
        // pto: %115
        ;
        int64_t v617 = v616 < v42 ? v42 : v616;
        // pto: %116
        ;
        const int64_t v618 = 0;
        // pto: %116
        ;
        __gm__ bfloat16_t* v619 = PTOAS__GLOBAL_TENSOR_DATA(v61);
        // pto: %116
        ;
        const int64_t v620 = 1;
        // pto: %116
        ;
        const int64_t v621 = 1;
        // pto: %116
        ;
        const int64_t v622 = 1;
        // pto: %116
        ;
        int64_t v623 = v404 * v33;
        // pto: %116
        ;
        int64_t v624 = v622 * v623;
        // pto: %116
        ;
        pto::Shape<1, 1, 1, -1, 64> v625 = pto::Shape<1, 1, 1, -1, 64>(v620, v621, v622, v404, v36);
        // pto: %116
        ;
        pto::Stride<-1, -1, -1, -1, -1> v626 = pto::Stride<-1, -1, -1, -1, -1>(v621 * v624, v624, v623, v33, v34);
        // pto: %116
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v627 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v619 + (v618 + v415 * v33 + v617 * v34), v625, v626);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
        TSTORE(v627, v612);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      };
      set_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
    };
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}