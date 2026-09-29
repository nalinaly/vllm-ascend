# SPMD源码清点

同名任务可能有ND/NZ或运行时互斥分支；以下按源码声明列出，不等于本次实测的任务数。
实际worker数和核时另见RESULTS及summary中的官方泳道。

| 文件 / 函数 | 任务 | 单组worker表达式 | 原sync_start |
| --- | --- | --- | --- |
| decode_compressor_ratio4.py:121 / compressor_ratio4_project | 'kv_score_proj' | `KV_SCORE_WORKERS` | False |
| decode_compressor_ratio4.py:182 / compressor_ratio4_pool_projected | 'scatter_softmax_pool' | `pool_workers` | False |
| decode_compressor_ratio4.py:325 / compressor_ratio4_cache_write | 'compress_state_commit' | `commit_workers` | False |
| decode_compressor_ratio4.py:351 / compressor_ratio4_cache_write | 'rmsnorm_rope_cache_write' | `rms_blocks` | False |
| decode_csa.py:262 / _decode_csa_tp1_layer | 'hc_widen_rms' | `pl.min(widen_rows, HC_WIDEN_WORKERS)` | False |
| decode_csa.py:318 / _decode_csa_tp1_layer | 'csa_rope_sign' | `pl.min(rope_sign_blocks, CSA_ROPE_WORKERS)` | False |
| decode_csa.py:363 / _decode_csa_tp1_layer | 'csa_cache_writeback' | `TP1_CSA_WB_WORKERS` | False |
| decode_indexer.py:597 / indexer_head_coefficients | 'indexer_head_coefficients' | `coefficient_workers` | False |
| decode_indexer.py:681 / indexer_score_topk_native_cube | 'indexer_score_topk_native_pair' | `TOPK_SCORE_WORKERS` | score_sync_start |
| decode_indexer.py:1495 / indexer_score_topk_forest | 'indexer_score_topk_leaf' | `TOPK_SCORE_WORKERS` | False |
| decode_indexer.py:1630 / indexer_score_topk_forest | 'indexer_topk_single_leaf_publish' | `TOPK_QUERY_WORKERS` | False |
| decode_indexer.py:1640 / indexer_score_topk_forest | 'indexer_topk_query_merge' | `TOPK_QUERY_WORKERS` | False |
| decode_indexer.py:1645 / indexer_score_topk_forest | 'indexer_topk_query_merge' | `TOPK_QUERY_WORKERS` | False |
| decode_indexer.py:1690 / indexer_qr_rope | 'idx_qr_dequant_rope' | `dq_rope_workers` | False |
| decode_indexer.py:1670 / indexer_qr_rope | 'idx_qr_proj_matmul' | `QR_PROJ_WORKERS` | False |
| decode_indexer.py:1787 / indexer_qr_hadamard_mm | 'qr_hadamard_matmul' | `QH_WORKERS` | False |
| decode_indexer.py:1801 / indexer_qr_hadamard_mm | 'qr_hadamard_quant' | `QH_QUANT_WORKERS` | False |
| decode_indexer.py:1885 / indexer_weights_project | 'weights_proj' | `weights_workers` | False |
| decode_indexer.py:1907 / indexer_weights_project | 'weights_proj_reduce' | `row_blocks` | False |
| decode_indexer_compressor.py:131 / indexer_compressor_project | 'kv_score_proj' | `KV_SCORE_WORKERS` | False |
| decode_indexer_compressor.py:189 / indexer_compressor_pool_projected | 'scatter_softmax_pool' | `pool_workers` | False |
| decode_indexer_compressor.py:265 / indexer_compressor_pool_projected | 'compress_state_commit' | `commit_workers` | False |
| decode_indexer_compressor.py:293 / indexer_compressor_pool_projected | 'indexer_boundary_init' | `b_dim` | False |
| decode_indexer_compressor.py:300 / indexer_compressor_pool_projected | 'rmsnorm_rope' | `rms_workers` | False |
| decode_indexer_compressor.py:440 / indexer_compressor_write | 'kv_and_cache_write' | `rms_blocks` | False |
| decode_o_proj.py:188 / _proj_a_mm_nz | 'proj_a_mm' | `proj_a_rows * (O_LORA // A_COL_TILE)` | False |
| decode_o_proj.py:234 / _proj_a_mm_nd | 'proj_a_mm' | `proj_a_rows * (O_LORA // A_COL_TILE)` | False |
| decode_o_proj.py:280 / _proj_b_mm_nd | 'proj_b_mm' | `proj_b_t_rows * (D // PROJ_B_D_TILE)` | False |
| decode_o_proj.py:364 / _proj_b_mm_nz | 'proj_b_mm' | `D // PROJ_B_D_TILE` | False |
| decode_o_proj.py:460 / _decode_o_proj_tp1_tiled | 'proj_b_act_hc_post' | `act_t_blks * (D // PROJ_B_ACT_N_TILE)` | False |
| decode_o_proj.py:420 / _decode_o_proj_tp1_tiled | 'quant' | `(t_dim + QUANT_TASK_T_TILE - 1) // QUANT_TASK_T_TILE` | False |
| decode_sparse_attn_csa.py:207 / sparse_attn_csa | 'csa_slots_build_valid_qk_plan' | `CSA_PLAN_WORKERS` | False |
| decode_sparse_attn_csa.py:350 / sparse_attn_csa | 'rope_cs' | `pl.min(rope_cs_blocks, ROPE_CS_WORKERS)` | False |
| decode_sparse_attn_csa.py:389 / sparse_attn_csa | 'qk_pv' | `NUM_QK_CORES` | False |
| qkv_proj_rope.py:152 / rope_prepare | 'q_rope_prepare' | `pl.min(Q_ROPE_WORKERS, token_tiles)` | False |
| qkv_proj_rope.py:248 / _q_proj_qa_nd | 'qr_proj_matmul' | `QR_N_BLOCKS * QR_OK * qr_m_groups` | False |
| qkv_proj_rope.py:316 / _q_proj_qa_nz | 'qr_proj_matmul' | `QR_N_BLOCKS * QR_OK * qr_m_groups` | False |
| qkv_proj_rope.py:434 / q_proj_qr_normalize | 'qr_rms_norm_quant' | `qr_norm_workers` | False |
| qkv_proj_rope.py:529 / q_proj_q_dequant | 'qproj_dequant_rms_nope_rope' | `Q_DEQUANT_WORKERS` | False |
| qkv_proj_rope.py:807 / _kv_project | 'kv_proj_matmul' | `HEAD_DIM // KV_N_TILE * KV_OK * kv_m_groups` | False |
| qkv_proj_rope.py:894 / kv_proj_rope | 'kv_rms_norm_rope' | `kv_token_tiles` | True |
| hc_pre.py:96 / hc_pre_gates_from_rms | 'hc_pre_linear' | `linear_workers` | False |
| hc_pre.py:114 / hc_pre_gates_from_rms | 'hc_pre_linear_reduce' | `t_linear // LINEAR_T_TILE` | False |
| hc_pre.py:129 / hc_pre_gates_from_rms | 'split_pre_post' | `pl.min(token_tiles, PRE_POST_WORKERS)` | False |
| hc_pre.py:157 / hc_pre_gates_from_rms | 'comb_sinkhorn' | `token_tiles` | False |
| hc_pre.py:300 / hc_pre_gates | 'hc_pre_rms' | `token_tiles` | False |
| hc_pre.py:348 / _hc_pre | 'mix_x' | `token_tiles * (D // D_SPMD)` | False |
| hc_pre.py:402 / hc_mix_norm | 'mix_x_rms_norm' | `t_pad // T_TILE` | False |
| q_projection.py:53 / _q_proj_q_matmul_nd | 'qproj_matmul' | `QPROJ_WORKERS` | False |
| q_projection.py:93 / _q_proj_q_matmul_nz | 'qproj_matmul' | `QPROJ_WORKERS` | False |

每组上限依运行时core type区分：纯AIV为48，AIC或MIX为24。原有多组并行要看每组，不按泳道总worker数套用此限制。
后续扩核必须重新计算真实工作份数、每份字节/算术量、重复权重读取以及独占写区；空worker不计为有效用核。
