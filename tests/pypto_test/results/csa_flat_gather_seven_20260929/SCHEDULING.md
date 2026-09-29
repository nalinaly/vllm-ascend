# 七档调度下一步依据：固定window_3

只分析已有level4泳道，不新增占卡。所有时刻按该窗口首个Worker start归零，单位μs。
end→FIN为逻辑任务末核结束到最后结束通知；FIN→dispatch可因early为负。
前置有无时戳的dummy则ready不可确认，不把部分前置当完整ready。
dispatch→start和启动分散含依赖门控/多波/竞争，均不是纯AICPU软件开销。
同名多行是不同逻辑任务（如O投影各组），原始task_id与直接前置见scheduling.json。
独立DFX不能与正式CSA事件相减，也不用于解释未同时profiling的P95样本。

## 128K/B4

| Task | 首start | 末end | end→FIN | 最晚直接前置FIN | FIN→dispatch | dispatch→start | early |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| qr_proj_matmul | 64.22 | 83.80 | 4.74 | mix_x_rms_norm | -16.88 | 17.84 | full |
| qr_rms_norm_quant | 89.68 | 95.36 | 2.90 | qr_proj_matmul | -17.84 | 18.98 | full |
| qproj_matmul | 112.70 | 169.02 | 5.12 | 未计时前置，无法确认 | — | 1.02 | unverifiable |
| qproj_dequant_rms_nope_rope | 179.96 | 200.82 | 6.50 | qproj_matmul | 5.06 | 0.76 | none |
| idx_qr_proj_matmul | 104.76 | 130.50 | 8.46 | qr_rms_norm_quant | 5.56 | 0.94 | none |
| idx_qr_dequant_rope | 147.56 | 166.66 | 8.34 | idx_qr_proj_matmul | 7.64 | 0.96 | none |
| qr_hadamard_matmul | 175.76 | 187.62 | 8.32 | idx_qr_dequant_rope | -11.78 | 12.54 | partial |
| qr_hadamard_quant | 196.70 | 213.24 | 2.48 | qr_hadamard_matmul | -6.02 | 6.78 | partial |
| indexer_head_coefficients | 216.52 | 218.26 | 9.04 | qr_hadamard_quant | -3.96 | 4.76 | full |
| indexer_score_topk_native_pair_aic | 227.94 | 307.48 | 3.50 | indexer_head_coefficients | -2.66 | 3.30 | full |
| indexer_topk_query_merge | 316.70 | 329.94 | 3.78 | indexer_score_topk_native_pair_aic | 4.54 | 1.18 | none |
| csa_slots_build_valid_qk_plan | 334.38 | 342.86 | 8.20 | indexer_topk_query_merge | -13.64 | 14.30 | full |
| qk_pv_aic | 351.66 | 407.54 | 3.12 | csa_slots_build_valid_qk_plan | -15.44 | 16.04 | full |
| proj_a_mm | 412.84 | 445.18 | 5.32 | qk_pv_aic | -47.18 | 49.36 | partial |
| proj_a_mm | 427.74 | 447.44 | 7.38 | qk_pv_aic | 8.46 | 8.62 | none |
| proj_a_mm | 426.72 | 461.10 | 4.10 | qk_pv_aic | 11.96 | 4.10 | none |
| proj_a_mm | 412.18 | 445.96 | 3.58 | qk_pv_aic | -47.82 | 49.34 | partial |
| proj_a_mm | 443.60 | 462.06 | 9.30 | qk_pv_aic | 18.32 | 14.62 | none |
| proj_a_mm | 428.34 | 461.30 | 2.68 | qk_pv_aic | 14.12 | 3.56 | none |
| proj_a_mm | 411.46 | 428.68 | 7.26 | qk_pv_aic | -54.14 | 54.94 | full |
| proj_a_mm | 414.70 | 431.56 | 6.44 | qk_pv_aic | -54.40 | 58.44 | full |
| quant | 437.36 | 444.18 | 0.38 | proj_a_mm | -20.32 | 21.74 | full |
| quant | 438.96 | 444.44 | 0.60 | proj_a_mm | -21.28 | 22.24 | full |
| quant | 472.08 | 477.70 | 2.42 | proj_a_mm | -29.30 | 30.02 | full |
| quant | 450.16 | 456.76 | 3.28 | proj_a_mm | -17.72 | 18.34 | full |
| quant | 451.40 | 457.44 | 4.52 | proj_a_mm | -17.72 | 18.62 | full |
| quant | 455.36 | 461.10 | 2.38 | proj_a_mm | -21.02 | 21.56 | full |
| quant | 464.92 | 470.56 | 3.68 | proj_a_mm | -23.26 | 24.20 | full |
| quant | 466.54 | 472.14 | 3.72 | proj_a_mm | -23.80 | 25.14 | full |
| proj_b_mm | 462.48 | 471.56 | 10.52 | quant | -9.24 | 9.76 | full |
| proj_b_mm | 468.02 | 478.46 | 1.02 | quant | -2.04 | 6.58 | partial |
| proj_b_mm | 480.64 | 488.36 | 1.70 | quant | -4.92 | 5.44 | full |
| proj_b_mm | 476.36 | 485.56 | 2.50 | quant | -8.04 | 8.54 | partial |
| proj_b_mm | 446.50 | 458.78 | 6.32 | quant | -2.04 | 3.98 | full |
| proj_b_mm | 460.56 | 469.02 | 3.06 | quant | -8.36 | 8.88 | full |
| proj_b_mm | 459.24 | 470.32 | 3.04 | quant | 3.08 | 11.12 | none |
| proj_b_mm | 475.16 | 482.84 | 3.96 | quant | -7.26 | 8.18 | partial |
| proj_b_act_hc_post | 490.76 | 515.92 | 1.36 | proj_b_mm | -6.24 | 6.94 | full |

[固定原始窗口](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_flat_gather_seven_20260929/h131072_b4/swimlane/candidate/dfx/window_3/merged_swimlane.json)

## 128K/B8

| Task | 首start | 末end | end→FIN | 最晚直接前置FIN | FIN→dispatch | dispatch→start | early |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| qr_proj_matmul | 71.96 | 93.84 | 5.14 | mix_x_rms_norm | -18.72 | 19.78 | full |
| qr_rms_norm_quant | 99.72 | 105.78 | 7.56 | qr_proj_matmul | -17.62 | 18.36 | full |
| qproj_matmul | 129.60 | 194.38 | 18.44 | 未计时前置，无法确认 | — | 4.50 | unverifiable |
| qproj_dequant_rms_nope_rope | 217.60 | 242.60 | 8.82 | qproj_matmul | 4.04 | 0.74 | none |
| idx_qr_proj_matmul | 120.10 | 170.50 | 1.68 | qr_rms_norm_quant | -6.62 | 13.38 | partial |
| idx_qr_dequant_rope | 178.36 | 201.98 | 17.36 | idx_qr_proj_matmul | 5.32 | 0.86 | none |
| qr_hadamard_matmul | 220.20 | 233.98 | 7.92 | idx_qr_dequant_rope | -23.62 | 24.48 | partial |
| qr_hadamard_quant | 242.70 | 255.48 | 4.80 | qr_hadamard_matmul | -9.68 | 10.48 | partial |
| indexer_head_coefficients | 260.82 | 263.04 | 11.60 | qr_hadamard_quant | -5.52 | 6.06 | full |
| indexer_score_topk_native_pair_aic | 275.46 | 398.76 | 4.52 | indexer_head_coefficients | -4.56 | 5.38 | full |
| indexer_topk_query_merge | 408.84 | 422.68 | 3.06 | indexer_score_topk_native_pair_aic | 4.64 | 0.92 | none |
| csa_slots_build_valid_qk_plan | 426.76 | 436.90 | 4.90 | indexer_topk_query_merge | -13.22 | 14.24 | full |
| qk_pv_aic | 442.54 | 526.94 | 3.80 | csa_slots_build_valid_qk_plan | -13.94 | 14.68 | full |
| proj_a_mm | 532.24 | 586.36 | 8.80 | qk_pv_aic | -75.84 | 77.34 | partial |
| proj_a_mm | 556.82 | 586.68 | 9.88 | qk_pv_aic | 14.50 | 11.58 | none |
| proj_a_mm | 533.46 | 561.20 | 10.84 | qk_pv_aic | -75.06 | 77.78 | full |
| proj_a_mm | 559.62 | 587.28 | 6.58 | qk_pv_aic | 9.00 | 19.88 | none |
| proj_a_mm | 585.42 | 611.98 | 5.76 | qk_pv_aic | 30.76 | 23.92 | none |
| proj_a_mm | 580.74 | 608.54 | 7.06 | qk_pv_aic | 34.12 | 15.88 | none |
| proj_a_mm | 534.08 | 560.88 | 8.54 | qk_pv_aic | -81.14 | 84.48 | full |
| proj_a_mm | 531.46 | 581.90 | 7.82 | qk_pv_aic | -82.06 | 82.78 | partial |
| quant | 616.62 | 622.54 | 7.44 | proj_a_mm | -44.70 | 45.72 | full |
| quant | 595.74 | 604.88 | 1.32 | proj_a_mm | -42.44 | 43.02 | full |
| quant | 572.64 | 582.26 | 3.96 | proj_a_mm | -28.52 | 29.12 | full |
| quant | 618.32 | 624.36 | 2.98 | proj_a_mm | -42.06 | 42.64 | full |
| quant | 570.16 | 580.66 | 0.80 | proj_a_mm | -24.74 | 25.48 | full |
| quant | 590.88 | 602.08 | 2.00 | proj_a_mm | -43.60 | 44.76 | full |
| quant | 594.48 | 604.18 | 1.74 | proj_a_mm | -46.44 | 47.06 | full |
| quant | 597.16 | 604.02 | 4.72 | proj_a_mm | -30.72 | 31.32 | full |
| proj_b_mm | 583.94 | 621.70 | 10.84 | quant | -8.32 | 10.80 | partial |
| proj_b_mm | 604.76 | 623.74 | 3.04 | quant | -12.64 | 13.32 | full |
| proj_b_mm | 618.78 | 632.78 | 8.48 | quant | 9.90 | 2.68 | none |
| proj_b_mm | 631.62 | 644.26 | 0.78 | quant | -3.10 | 7.38 | full |
| proj_b_mm | 606.52 | 626.22 | 2.82 | quant | -7.66 | 8.26 | full |
| proj_b_mm | 630.54 | 642.08 | 2.12 | quant | -10.62 | 11.18 | full |
| proj_b_mm | 586.92 | 622.16 | 3.02 | quant | -11.76 | 12.46 | partial |
| proj_b_mm | 622.68 | 635.28 | 1.90 | quant | 12.48 | 1.46 | none |
| proj_b_act_hc_post | 645.72 | 671.86 | 2.84 | proj_b_mm | -11.14 | 11.82 | full |

[固定原始窗口](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_flat_gather_seven_20260929/h131072_b8/swimlane/candidate/dfx/window_3/merged_swimlane.json)

## 128K/B16

| Task | 首start | 末end | end→FIN | 最晚直接前置FIN | FIN→dispatch | dispatch→start | early |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| qr_proj_matmul | 85.82 | 101.68 | 9.46 | mix_x_rms_norm | -21.36 | 22.40 | full |
| qr_rms_norm_quant | 112.02 | 119.82 | 10.58 | qr_proj_matmul | -3.96 | 4.84 | full |
| qproj_matmul | 144.48 | 229.80 | 0.52 | 未计时前置，无法确认 | — | 0.88 | unverifiable |
| qproj_dequant_rms_nope_rope | 234.44 | 264.70 | 17.00 | qproj_matmul | 3.18 | 0.94 | none |
| idx_qr_proj_matmul | 140.20 | 232.18 | 7.60 | qr_rms_norm_quant | 8.36 | 1.44 | none |
| idx_qr_dequant_rope | 252.82 | 278.46 | 6.44 | idx_qr_proj_matmul | 7.34 | 5.70 | none |
| qr_hadamard_matmul | 285.44 | 291.82 | 7.34 | idx_qr_dequant_rope | -16.80 | 17.34 | full |
| qr_hadamard_quant | 300.00 | 320.80 | 3.60 | qr_hadamard_matmul | -9.74 | 10.58 | full |
| indexer_head_coefficients | 325.20 | 329.08 | 13.12 | qr_hadamard_quant | -21.46 | 22.26 | full |
| indexer_score_topk_native_pair_aic | 343.18 | 578.30 | 0.34 | indexer_head_coefficients | -5.26 | 6.24 | full |
| indexer_topk_query_merge | 584.30 | 597.46 | 2.96 | indexer_score_topk_native_pair_aic | 4.66 | 1.00 | none |
| csa_slots_build_valid_qk_plan | 601.04 | 609.88 | 5.36 | indexer_topk_query_merge | -9.66 | 10.28 | full |
| qk_pv_aic | 615.98 | 771.00 | 4.44 | csa_slots_build_valid_qk_plan | -13.14 | 13.88 | full |
| proj_a_mm | 776.10 | 838.00 | 4.40 | qk_pv_aic | -155.66 | 156.32 | partial |
| proj_a_mm | 835.02 | 861.94 | 7.90 | qk_pv_aic | 34.36 | 25.22 | none |
| proj_a_mm | 776.94 | 808.66 | 12.46 | qk_pv_aic | -153.14 | 154.64 | full |
| proj_a_mm | 778.16 | 838.98 | 5.22 | qk_pv_aic | -148.08 | 150.80 | partial |
| proj_a_mm | 808.32 | 862.46 | 1.90 | qk_pv_aic | 18.04 | 14.84 | none |
| proj_a_mm | 777.66 | 809.26 | 8.08 | qk_pv_aic | -154.72 | 156.94 | full |
| proj_a_mm | 805.58 | 838.72 | 5.76 | qk_pv_aic | 12.40 | 17.74 | none |
| proj_a_mm | 808.16 | 861.62 | 4.96 | qk_pv_aic | 21.10 | 11.62 | none |
| quant | 821.78 | 831.44 | 1.64 | proj_a_mm | -40.04 | 40.70 | full |
| quant | 845.32 | 854.62 | 1.14 | proj_a_mm | -26.30 | 27.42 | full |
| quant | 843.52 | 853.62 | 2.86 | proj_a_mm | -56.82 | 57.94 | full |
| quant | 817.92 | 829.38 | 4.34 | proj_a_mm | -34.12 | 34.70 | full |
| quant | 845.00 | 854.94 | 2.58 | proj_a_mm | -23.14 | 23.66 | full |
| quant | 865.70 | 872.42 | 5.84 | proj_a_mm | -38.10 | 39.44 | full |
| quant | 870.38 | 876.58 | 5.70 | proj_a_mm | -44.80 | 45.34 | full |
| quant | 867.18 | 874.22 | 2.24 | proj_a_mm | -43.26 | 43.86 | full |
| proj_b_mm | 853.32 | 874.78 | 2.82 | quant | 3.62 | 16.62 | none |
| proj_b_mm | 865.22 | 879.50 | 13.24 | quant | 4.58 | 4.88 | none |
| proj_b_mm | 883.00 | 893.46 | 1.08 | quant | -7.56 | 8.28 | full |
| proj_b_mm | 838.24 | 853.64 | 6.38 | quant | -8.62 | 13.14 | full |
| proj_b_mm | 858.20 | 874.22 | 0.98 | quant | -7.24 | 7.92 | full |
| proj_b_mm | 879.00 | 889.80 | 1.36 | quant | -11.66 | 12.40 | full |
| proj_b_mm | 857.20 | 874.48 | 7.86 | quant | -10.04 | 10.76 | full |
| proj_b_mm | 877.08 | 888.10 | 4.04 | quant | -6.44 | 7.06 | full |
| proj_b_act_hc_post | 895.28 | 924.64 | 2.34 | proj_b_mm | -9.98 | 10.72 | full |

[固定原始窗口](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_sparse_rope_flat_gather_20260929/h131072_b16/swimlane/candidate/dfx/window_3/merged_swimlane.json)

## 128K/B24

| Task | 首start | 末end | end→FIN | 最晚直接前置FIN | FIN→dispatch | dispatch→start | early |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| qr_proj_matmul | 103.08 | 126.98 | 3.62 | mix_x_rms_norm | -21.92 | 22.80 | full |
| qr_rms_norm_quant | 131.30 | 142.46 | 4.04 | qr_proj_matmul | -25.42 | 26.12 | full |
| qproj_matmul | 192.86 | 289.08 | 7.38 | 未计时前置，无法确认 | — | 4.68 | unverifiable |
| qproj_dequant_rms_nope_rope | 315.12 | 353.66 | 6.90 | qproj_matmul | 5.88 | 12.78 | none |
| idx_qr_proj_matmul | 174.38 | 219.76 | 2.42 | qr_rms_norm_quant | -0.64 | 28.52 | partial |
| idx_qr_dequant_rope | 226.04 | 282.30 | 0.58 | idx_qr_proj_matmul | 2.98 | 0.88 | none |
| qr_hadamard_matmul | 283.46 | 294.62 | 2.90 | idx_qr_dequant_rope | -14.94 | 15.52 | full |
| qr_hadamard_quant | 298.24 | 318.92 | 11.80 | qr_hadamard_matmul | -10.10 | 10.82 | full |
| indexer_head_coefficients | 346.58 | 355.14 | 5.42 | qr_hadamard_quant | -5.28 | 21.14 | full |
| indexer_score_topk_native_pair_aic | 370.24 | 714.10 | 1.12 | indexer_head_coefficients | 6.84 | 2.84 | none |
| indexer_topk_query_merge | 721.16 | 738.34 | 3.08 | indexer_score_topk_native_pair_aic | 4.84 | 1.10 | none |
| csa_slots_build_valid_qk_plan | 742.08 | 754.16 | 4.58 | indexer_topk_query_merge | -16.96 | 17.62 | full |
| qk_pv_aic | 759.42 | 985.62 | 4.72 | csa_slots_build_valid_qk_plan | -15.54 | 16.22 | full |
| proj_a_mm | 990.92 | 1023.82 | 3.34 | qk_pv_aic | -225.00 | 225.58 | full |
| proj_a_mm | 1020.94 | 1070.64 | 2.58 | qk_pv_aic | 23.44 | 7.16 | none |
| proj_a_mm | 1045.34 | 1080.00 | 2.88 | qk_pv_aic | 33.28 | 21.72 | none |
| proj_a_mm | 992.30 | 1022.92 | 5.66 | qk_pv_aic | -214.94 | 216.90 | full |
| proj_a_mm | 1020.36 | 1052.68 | 4.46 | qk_pv_aic | 8.66 | 21.36 | none |
| proj_a_mm | 991.68 | 1022.70 | 6.52 | qk_pv_aic | -220.44 | 221.78 | full |
| proj_a_mm | 1023.50 | 1078.14 | 4.50 | qk_pv_aic | 12.84 | 20.32 | none |
| proj_a_mm | 1021.20 | 1051.34 | 4.30 | qk_pv_aic | 2.20 | 28.66 | none |
| quant | 1027.72 | 1036.10 | 4.70 | proj_a_mm | -27.28 | 27.84 | full |
| quant | 1073.82 | 1081.40 | 8.28 | proj_a_mm | -40.58 | 41.18 | full |
| quant | 1057.72 | 1065.32 | 3.28 | proj_a_mm | -34.18 | 34.76 | full |
| quant | 1029.78 | 1038.54 | 5.10 | proj_a_mm | -27.00 | 27.56 | full |
| quant | 1083.52 | 1091.94 | 1.18 | proj_a_mm | -48.08 | 48.72 | full |
| quant | 1056.30 | 1063.96 | 0.76 | proj_a_mm | -48.16 | 48.82 | full |
| quant | 1083.24 | 1092.42 | 3.58 | proj_a_mm | -49.20 | 49.80 | full |
| quant | 1029.38 | 1039.18 | 6.66 | proj_a_mm | -23.14 | 23.94 | full |
| proj_b_mm | 1070.46 | 1098.26 | 7.78 | quant | 8.22 | 16.40 | none |
| proj_b_mm | 1091.00 | 1115.68 | 0.48 | quant | -13.92 | 15.24 | full |
| proj_b_mm | 1046.54 | 1079.48 | 9.62 | quant | -5.00 | 10.74 | full |
| proj_b_mm | 1068.12 | 1101.00 | 3.64 | quant | -6.12 | 9.52 | full |
| proj_b_mm | 1096.68 | 1124.08 | 1.08 | quant | -10.60 | 11.28 | full |
| proj_b_mm | 1085.38 | 1109.60 | 3.40 | quant | 3.72 | 13.06 | none |
| proj_b_mm | 1101.26 | 1129.26 | 0.20 | quant | -1.84 | 9.98 | partial |
| proj_b_mm | 1066.42 | 1096.62 | 2.42 | quant | 5.26 | 17.52 | none |
| proj_b_act_hc_post | 1130.02 | 1164.04 | 2.76 | proj_b_mm | -30.74 | 31.30 | full |

[固定原始窗口](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_flat_gather_seven_20260929/h131072_b24/swimlane/candidate/dfx/window_3/merged_swimlane.json)

## 8K/B16

| Task | 首start | 末end | end→FIN | 最晚直接前置FIN | FIN→dispatch | dispatch→start | early |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| qr_proj_matmul | 85.40 | 101.48 | 6.38 | mix_x_rms_norm | -20.28 | 21.36 | full |
| qr_rms_norm_quant | 108.80 | 116.48 | 6.30 | qr_proj_matmul | -17.94 | 18.88 | full |
| qproj_matmul | 137.42 | 232.00 | 1.76 | 未计时前置，无法确认 | — | 10.02 | unverifiable |
| qproj_dequant_rms_nope_rope | 238.24 | 275.76 | 4.74 | qproj_matmul | 3.58 | 0.90 | none |
| idx_qr_proj_matmul | 127.16 | 233.52 | 6.66 | qr_rms_norm_quant | -9.66 | 14.04 | partial |
| idx_qr_dequant_rope | 257.64 | 291.36 | 1.48 | idx_qr_proj_matmul | 12.98 | 4.48 | none |
| qr_hadamard_matmul | 293.36 | 300.26 | 6.10 | idx_qr_dequant_rope | -10.68 | 11.20 | full |
| qr_hadamard_quant | 307.08 | 325.30 | 5.10 | qr_hadamard_matmul | -11.78 | 12.50 | full |
| indexer_head_coefficients | 331.04 | 340.18 | 8.48 | qr_hadamard_quant | -21.02 | 21.66 | full |
| indexer_score_topk_native_pair_aic | 349.44 | 404.38 | 2.76 | indexer_head_coefficients | -7.62 | 8.40 | partial |
| indexer_topk_query_merge | 407.84 | 419.42 | 2.72 | indexer_score_topk_native_pair_aic | -47.58 | 48.28 | full |
| csa_slots_build_valid_qk_plan | 422.86 | 432.54 | 8.08 | indexer_topk_query_merge | -11.60 | 12.32 | full |
| qk_pv_aic | 441.26 | 574.58 | 3.22 | csa_slots_build_valid_qk_plan | -16.44 | 17.08 | full |
| proj_a_mm | 582.52 | 614.12 | 17.60 | qk_pv_aic | -131.78 | 136.50 | full |
| proj_a_mm | 609.48 | 642.80 | 10.34 | qk_pv_aic | 9.90 | 21.78 | none |
| proj_a_mm | 610.70 | 667.32 | 7.28 | qk_pv_aic | 15.48 | 17.42 | none |
| proj_a_mm | 580.30 | 640.94 | 2.80 | qk_pv_aic | -123.04 | 125.54 | partial |
| proj_a_mm | 579.12 | 638.30 | 5.46 | qk_pv_aic | -124.58 | 125.90 | partial |
| proj_a_mm | 581.04 | 662.56 | 6.66 | qk_pv_aic | -121.38 | 124.62 | partial |
| proj_a_mm | 578.46 | 608.78 | 3.66 | qk_pv_aic | -131.52 | 132.18 | full |
| proj_a_mm | 637.58 | 664.24 | 3.46 | qk_pv_aic | 37.04 | 22.74 | none |
| quant | 675.62 | 682.66 | 2.06 | proj_a_mm | -48.10 | 49.12 | full |
| quant | 617.10 | 629.14 | 3.12 | proj_a_mm | 2.38 | 2.28 | none |
| quant | 632.30 | 639.94 | 1.24 | proj_a_mm | -11.94 | 12.52 | full |
| quant | 669.74 | 679.42 | 3.84 | proj_a_mm | -46.78 | 47.30 | full |
| quant | 644.30 | 654.90 | 4.60 | proj_a_mm | -21.68 | 22.22 | full |
| quant | 668.26 | 679.30 | 6.16 | proj_a_mm | -43.20 | 43.76 | full |
| quant | 653.76 | 661.10 | 1.08 | proj_a_mm | -29.62 | 30.24 | full |
| quant | 644.34 | 653.18 | 10.32 | proj_a_mm | -18.84 | 19.44 | full |
| proj_b_mm | 664.20 | 684.12 | 17.68 | quant | -10.08 | 10.78 | full |
| proj_b_mm | 685.32 | 701.28 | 3.20 | quant | -8.18 | 8.78 | full |
| proj_b_mm | 641.12 | 658.44 | 6.70 | quant | -4.60 | 13.46 | partial |
| proj_b_mm | 660.48 | 682.84 | 3.66 | quant | -13.44 | 14.42 | full |
| proj_b_mm | 674.40 | 691.14 | 12.52 | quant | 1.18 | 11.04 | none |
| proj_b_mm | 685.70 | 709.18 | 0.82 | quant | -2.80 | 5.24 | partial |
| proj_b_mm | 662.40 | 679.56 | 7.20 | quant | -0.90 | 22.12 | partial |
| proj_b_mm | 685.98 | 701.84 | 1.06 | quant | -13.04 | 13.56 | full |
| proj_b_act_hc_post | 710.74 | 739.92 | 2.40 | proj_b_mm | -16.88 | 17.62 | full |

[固定原始窗口](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_flat_gather_seven_20260929/h8192_b16/swimlane/candidate/dfx/window_3/merged_swimlane.json)

## 8K/B24

| Task | 首start | 末end | end→FIN | 最晚直接前置FIN | FIN→dispatch | dispatch→start | early |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| qr_proj_matmul | 107.46 | 130.98 | 6.34 | mix_x_rms_norm | -22.40 | 23.62 | full |
| qr_rms_norm_quant | 138.16 | 149.40 | 23.28 | qr_proj_matmul | -25.62 | 26.46 | full |
| qproj_matmul | 195.54 | 300.24 | 3.84 | 未计时前置，无法确认 | — | 14.22 | unverifiable |
| qproj_dequant_rms_nope_rope | 309.04 | 370.22 | 2.58 | qproj_matmul | 4.10 | 0.86 | none |
| idx_qr_proj_matmul | 181.50 | 267.16 | 1.24 | qr_rms_norm_quant | -23.44 | 32.26 | partial |
| idx_qr_dequant_rope | 275.22 | 317.84 | 1.12 | idx_qr_proj_matmul | 5.98 | 0.84 | none |
| qr_hadamard_matmul | 319.62 | 326.38 | 10.32 | idx_qr_dequant_rope | -32.42 | 33.08 | full |
| qr_hadamard_quant | 337.46 | 361.34 | 0.74 | qr_hadamard_matmul | -10.44 | 11.20 | partial |
| indexer_head_coefficients | 362.66 | 372.48 | 1.46 | qr_hadamard_quant | -14.40 | 14.98 | full |
| indexer_score_topk_native_pair_aic | 374.72 | 436.62 | 1.28 | indexer_head_coefficients | -8.38 | 9.16 | partial |
| indexer_topk_query_merge | 438.52 | 449.92 | 2.74 | indexer_score_topk_native_pair_aic | -49.12 | 49.74 | full |
| csa_slots_build_valid_qk_plan | 453.40 | 465.30 | 3.04 | indexer_topk_query_merge | -11.10 | 11.84 | full |
| qk_pv_aic | 469.08 | 654.16 | 4.16 | csa_slots_build_valid_qk_plan | -13.88 | 14.62 | full |
| proj_a_mm | 661.88 | 693.24 | 16.38 | qk_pv_aic | -184.38 | 187.94 | full |
| proj_a_mm | 688.88 | 731.40 | 9.10 | qk_pv_aic | 13.34 | 17.22 | none |
| proj_a_mm | 726.14 | 765.86 | 9.66 | qk_pv_aic | 32.24 | 35.58 | none |
| proj_a_mm | 659.68 | 690.30 | 3.36 | qk_pv_aic | -178.54 | 179.90 | full |
| proj_a_mm | 688.44 | 732.44 | 4.50 | qk_pv_aic | 7.44 | 22.68 | none |
| proj_a_mm | 726.52 | 764.30 | 7.82 | qk_pv_aic | 35.08 | 33.12 | none |
| proj_a_mm | 659.00 | 691.70 | 4.26 | qk_pv_aic | -184.66 | 185.34 | full |
| proj_a_mm | 689.32 | 728.50 | 5.54 | qk_pv_aic | 4.38 | 26.62 | none |
| quant | 737.44 | 748.94 | 0.92 | proj_a_mm | -64.02 | 64.52 | full |
| quant | 696.58 | 708.36 | 7.46 | proj_a_mm | -28.74 | 29.36 | full |
| quant | 772.62 | 780.36 | 2.32 | proj_a_mm | -58.16 | 58.66 | full |
| quant | 734.88 | 747.18 | 5.02 | proj_a_mm | -37.40 | 38.24 | full |
| quant | 694.58 | 707.96 | 2.04 | proj_a_mm | -23.82 | 24.74 | full |
| quant | 776.08 | 783.94 | 1.38 | proj_a_mm | -58.78 | 59.34 | full |
| quant | 741.06 | 751.84 | 4.70 | proj_a_mm | -41.10 | 41.66 | full |
| quant | 710.30 | 722.24 | 1.32 | proj_a_mm | -38.06 | 38.74 | full |
| proj_b_mm | 755.10 | 789.86 | 7.02 | quant | 12.22 | 19.32 | none |
| proj_b_mm | 783.62 | 813.90 | 1.78 | quant | 13.64 | 13.44 | none |
| proj_b_mm | 762.12 | 789.52 | 12.28 | quant | 13.92 | 32.38 | none |
| proj_b_mm | 785.44 | 814.58 | 1.50 | quant | -8.94 | 11.70 | full |
| proj_b_mm | 754.56 | 789.66 | 7.94 | quant | -10.06 | 12.42 | full |
| proj_b_mm | 728.32 | 755.96 | 3.60 | quant | -11.26 | 29.58 | full |
| proj_b_mm | 780.48 | 806.14 | 4.52 | quant | 11.00 | 19.62 | none |
| proj_b_mm | 805.08 | 827.54 | 1.04 | quant | -0.34 | 20.10 | partial |
| proj_b_act_hc_post | 829.18 | 862.08 | 3.04 | proj_b_mm | -39.22 | 39.82 | full |

[固定原始窗口](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_sparse_rope_flat_gather_20260929/h8192_b24/swimlane/candidate/dfx/window_3/merged_swimlane.json)

## 8K/B32

| Task | 首start | 末end | end→FIN | 最晚直接前置FIN | FIN→dispatch | dispatch→start | early |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| qr_proj_matmul | 115.88 | 136.74 | 4.70 | mix_x_rms_norm | -23.50 | 24.60 | full |
| qr_rms_norm_quant | 142.26 | 153.56 | 2.42 | qr_proj_matmul | -19.20 | 20.02 | full |
| qproj_matmul | 189.90 | 322.20 | 3.20 | 未计时前置，无法确认 | — | 12.98 | unverifiable |
| qproj_dequant_rms_nope_rope | 339.92 | 405.88 | 4.98 | qproj_matmul | 13.58 | 0.94 | none |
| idx_qr_proj_matmul | 173.40 | 228.98 | 6.78 | qr_rms_norm_quant | 8.64 | 8.78 | none |
| idx_qr_dequant_rope | 240.16 | 281.70 | 4.26 | idx_qr_proj_matmul | 3.26 | 1.14 | none |
| qr_hadamard_matmul | 286.60 | 328.48 | 3.30 | idx_qr_dequant_rope | -25.60 | 26.24 | partial |
| qr_hadamard_quant | 332.28 | 395.86 | 0.60 | qr_hadamard_matmul | -41.02 | 41.52 | partial |
| indexer_head_coefficients | 397.04 | 408.00 | 8.40 | qr_hadamard_quant | -38.90 | 39.48 | full |
| indexer_score_topk_native_pair_aic | 417.18 | 505.70 | 1.80 | indexer_head_coefficients | -16.86 | 17.64 | partial |
| indexer_topk_query_merge | 508.26 | 522.22 | 2.78 | indexer_score_topk_native_pair_aic | -79.58 | 80.34 | full |
| csa_slots_build_valid_qk_plan | 525.80 | 538.82 | 3.60 | indexer_topk_query_merge | -13.86 | 14.66 | full |
| qk_pv_aic | 543.08 | 780.38 | 2.74 | csa_slots_build_valid_qk_plan | -15.32 | 15.98 | full |
| proj_a_mm | 787.28 | 842.38 | 6.28 | qk_pv_aic | -228.96 | 233.12 | partial |
| proj_a_mm | 822.68 | 884.98 | 1.94 | qk_pv_aic | 14.76 | 24.80 | none |
| proj_a_mm | 784.60 | 822.30 | 6.46 | qk_pv_aic | -235.26 | 236.74 | full |
| proj_a_mm | 819.92 | 863.00 | 7.42 | qk_pv_aic | 9.48 | 27.32 | none |
| proj_a_mm | 822.06 | 900.76 | 8.46 | qk_pv_aic | 14.14 | 24.80 | none |
| proj_a_mm | 783.86 | 822.78 | 4.16 | qk_pv_aic | -236.84 | 237.58 | full |
| proj_a_mm | 842.68 | 902.64 | 2.28 | qk_pv_aic | 46.62 | 12.94 | none |
| proj_a_mm | 785.54 | 863.06 | 7.56 | qk_pv_aic | -227.50 | 229.92 | partial |
| quant | 871.02 | 881.62 | 4.76 | proj_a_mm | -39.38 | 39.98 | full |
| quant | 905.72 | 915.88 | 6.34 | proj_a_mm | -65.20 | 66.00 | full |
| quant | 827.68 | 838.22 | 6.36 | proj_a_mm | -34.98 | 35.72 | full |
| quant | 849.86 | 861.64 | 6.64 | proj_a_mm | -51.26 | 52.46 | full |
| quant | 887.46 | 897.70 | 3.26 | proj_a_mm | -49.08 | 49.62 | full |
| quant | 829.44 | 840.08 | 7.06 | proj_a_mm | -33.78 | 34.46 | full |
| quant | 871.22 | 882.32 | 7.78 | proj_a_mm | -35.36 | 35.96 | full |
| quant | 909.68 | 919.96 | 6.60 | proj_a_mm | -64.20 | 64.66 | full |
| proj_b_mm | 864.12 | 925.58 | 8.30 | quant | -3.56 | 20.54 | partial |
| proj_b_mm | 891.32 | 936.54 | 3.90 | quant | -8.60 | 9.82 | partial |
| proj_b_mm | 922.88 | 951.68 | 0.82 | quant | -14.92 | 15.58 | full |
| proj_b_mm | 927.12 | 957.42 | 0.24 | quant | -13.08 | 13.64 | full |
| proj_b_mm | 862.98 | 910.26 | 3.26 | quant | -10.14 | 28.54 | partial |
| proj_b_mm | 891.40 | 937.42 | 1.42 | quant | -11.72 | 16.74 | partial |
| proj_b_mm | 916.76 | 948.08 | 1.80 | quant | -5.24 | 21.04 | partial |
| proj_b_mm | 885.80 | 925.64 | 9.34 | quant | -1.50 | 19.02 | partial |
| proj_b_act_hc_post | 958.24 | 995.96 | 2.48 | proj_b_mm | -29.40 | 29.98 | full |

[固定原始窗口](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_flat_gather_seven_20260929/h8192_b32/swimlane/candidate/dfx/window_3/merged_swimlane.json)

## 727.98μs历史上游组织参照

上游缺版本和Scheduler View，不能比较纯调度开销；此处使用当前8K/B16的固定window_3。
上游merge_norm独立、当前融合在qk_pv，比较的两段终点都在最终发布之后。

| 不重叠Worker分段 | 历史上游 | 当前8K/B16 |
| --- | ---: | ---: |
| 首receive→norm结束 | 66.62 | 82.04 |
| norm结束→Sparse首receive | 317.14 | 346.40 |
| Sparse首receive→最终发布 | 184.90 | 147.12 |
| 发布→末Worker | 159.32 | 165.34 |
| 总Worker窗口 | 727.98 | 740.90 |
