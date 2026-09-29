# CSA：最新七档核内差距与下一步

更新：2026-09-29。完整七档对应 **f4861832 / CANN9.2**；包含O-B激活L1复用、整行HC收尾融合与七处early。
七档统一冻结源码：128K B4/B8/B16/B24、8K B16/B24/B32。两代表档复用，其余五档补齐；
Native沿用已完成最新标准，均auto设备0但采样分属不同任务。这是现有结果对照，不能归因单项收益。

Native整体为npugraph_ex、dynamic=False/fullgraph=True、inplace_pass/static/SuperKernel开启；
核内诊断仅关闭SuperKernel。PTO保留自身custom-op图。两侧mode2/det0、PTO atomic0、EPLB关闭，
真实第二个CSA层权重与合成历史，完整HC_pre+norm+CSA+HC_post，5预热20次正式设备事件。
Native仍为release custom二进制；ops-transformer28f40354/ops-nn19614968/ops-math361722c0是源码参考。

[完整结果](results/csa_early_chain_seven_20260929/RESULTS.md)、[任务明细](results/csa_early_chain_seven_20260929/TASKS.md)、
[七份PTO泳道下载](results/csa_early_chain_seven_20260929/download_pto_swimlanes/README.md)。

## 完整CSA

| 档位 | Native均值μs | PTO均值μs | PTO变化 | Native/PTO P95μs |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4 | 748.298 | 642.806 | -14.098% | 751.100/656.580 |
| 128K/B8 | 859.563 | 738.032 | -14.139% | 861.620/748.320 |
| 128K/B16 | 1130.853 | 965.879 | -14.588% | 1135.580/986.840 |
| 128K/B24 | 1281.888 | 1240.775 | -3.207% | 1289.880/1262.700 |
| 8K/B16 | 757.482 | 753.921 | -0.470% | 760.120/769.420 |
| 8K/B24 | 915.371 | 917.581 | +0.241% | 920.340/933.760 |
| 8K/B32 | 1062.079 | 1064.999 | +0.275% | 1066.220/1088.760 |

各上下文内batch等权，128K -11.508%、8K +0.015%，8:2 -9.203%。
PTO P95/P50为1.0154–1.0259，max/P50最高1.0405；累计0/140超过各自P50的105%。
短档P95仍高于Native；不以这140次无大尖峰关闭历史间歇长尾或EP16问题。
七档各自图重放八类状态/保护区及28窗官方join通过；跨版本状态依据为两代表档与尾行/padding。
这些不替代Native/PTO精度、真实模型token/DSpark或EP16 forward验收。

## Indexer核内

Native独立QLI包含系数、Score和归并，PTO拆成独立任务；PMU与worker计时边界不同。
单位μs，PTO为四窗均值。以下差值用于定位，不是严格纯算术差或可回收时长。

| 档位 | Native QLI AIC/AIV | PTO Score AIC/AIV | PTO系数 | PTO merge |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4 | 66.363/64.369 | 67.824/74.492 | 1.695 | 4.780 |
| 128K/B8 | 126.316/126.021 | 118.887/123.043 | 2.296 | 6.358 |
| 128K/B16 | 242.293/241.730 | 231.594/234.632 | 2.589 | 8.326 |
| 128K/B24 | 376.467/376.063 | 328.287/335.803 | 4.198 | 10.960 |
| 8K/B16 | 39.630/39.050 | 41.932/46.313 | 4.953 | 7.936 |
| 8K/B24 | 56.051/55.666 | 29.710/43.371 | 2.610 | 11.271 |
| 8K/B32 | 68.901/68.365 | 80.880/85.184 | 9.555 | 10.752 |

长S6完整query单根，短B16/B32为双query、短B24为S6；系数/scale提交/归并仍独立。
主性能Native含QLI→Sparse SuperKernel，不将其拆成这里的独立核时。
[算术与输入差异](DSV4_FLASH_CSA_INDEXER_NATIVE_GAP.md)。

## Sparse attention

PTO已融合最终归一化、逆RoPE和发布；Native下列Sparse不含独立逆RoPE，不能机械相减。

| 档位 | Native Sparse AIC/AIV | PTO融合Sparse AIC/AIV |
| --- | ---: | ---: |
| 128K/B4 | 53.214/57.090 | 44.666/50.676 |
| 128K/B8 | 95.667/99.998 | 79.235/83.032 |
| 128K/B16 | 167.255/169.662 | 152.027/155.857 |
| 128K/B24 | 221.309/223.867 | 229.268/233.282 |
| 8K/B16 | 95.677/98.084 | 127.463/131.246 |
| 8K/B24 | 166.763/169.371 | 177.692/181.528 |
| 8K/B32 | 210.209/212.812 | 237.615/241.586 |

长B24及短档Sparse仍有明显参考差距；Native各pipeline可重叠，MAC/MTE/FIX/Vector计数不能相加。

## Q/O及融合收尾

128K/B16、B24四窗均值。启动分散包含资源争用与必要多波，不全是调度器软件开销。

| Task | B16 worker/核时μs/启动分散μs | B24 worker/核时μs/启动分散μs |
| --- | ---: | ---: |
| hc_widen_rms | 12/12.663/1.075 | 18/12.823/4.010 |
| hc_pre_linear | 24/7.758/5.515 | 24/9.535/6.375 |
| comb_sinkhorn | 12/15.343/0.275 | 18/15.265/3.355 |
| mix_x_rms_norm | 12/16.944/0.520 | 18/17.125/2.595 |
| qproj_matmul | 24/38.855/58.460 | 24/50.398/64.520 |
| qproj_dequant_rms_nope_rope | 48/21.220/16.330 | 48/34.039/15.750 |
| proj_a_mm | 64/28.693/71.020 | 64/26.588/61.460 |
| quant | 24/8.680/57.635 | 40/7.004/57.465 |
| proj_b_mm | 64/10.596/52.170 | 64/18.994/64.105 |
| proj_b_act_hc_post | 24/27.021/0.460 | 36/33.662/0.740 |

当前没有独立hc_post，proj_b_act_hc_post合并O-B反量化与HC算术，保留BF16往返和加法顺序。
Q_B的24份工作与Compressor/Indexer投影争用AIC，不能据此认为20份必定一波完成。

## 已采用策略及下一步

1. [O-B AL1复用/K128双缓冲](results/csa_ob_activation_l1_k128_20260929/README.md)：参考最新ops-nn，
   局部A/B长短B16核内−5.720%/+5.852%，8:2−3.406%，完整CSA−1.296%。短档核内代价保留。
2. [整D单行收尾融合](results/csa_ob_hc_scalar_fused_20260929/README.md)：两代表档收尾总核时−10.970%/−8.642%，
   CSA−1.092%/−0.384%，状态及尾行通过。移除GM交接和一级任务，不是关闭early的收益。
3. [HCA early借鉴](results/csa_indexer_early_chain_20260929/README.md)：七处标志纯调度，
   CSA长B16−0.728%、短B24−1.872%，8:2−0.957%；长档P95+1.980μs，Score DFX核时约+4.4%单列。
   cache提交提前约39.5μs而Score首start几乎不变，剩余关键前置指向query链。
4. [Q_B 24→20候选](results/csa_qb_workers20_20260929/README.md)已完成，默认不采用：
   CSA长+2.337%、短+3.465%，8:2 +2.563%，P95均升。Q_B总核时8:2 −0.803%但四窗重叠，
   长短跨度+3.376%/+18.337%，20份仍落16–19个物理核，未形成HCA那样的收益。
   随后[Q反量化双head合批](results/csa_qdequant_pair_20260929/README.md)也不采用：
   目标核长+12.492%、短+20.088%，8:2 +14.012%；CSA 8:2 +1.179%，八类状态/16窗通过。
   原gather仍逐行处理，去掉内层TCONCAT和减少循环次数并未带来设备收益。
   [整块Gather](results/csa_qrope_flat_gather_20260929/README.md)已采用，保持单head和48-worker：
   参考AscendC整块索引，将8×64变为一次1×512 tile.gather；CPU确认无逐行TMOV。
   目标核长−3.980%、短−1.821%，8:2 −3.548%，但四窗范围重叠；CSA 8:2 −0.754%，P95/max均降。
   八类完整状态、16窗及H127/B3/padding通过，仅性能版单文件。本页七档表尚未包含此最新局部改动。
5. 保留长小batch Score、长B24与短档Sparse的核内入口；128K优先、长短8:2，真实核内收益保留，
   明显场景分化在同一算子内部按输入选择。两版Sparse计划提前已否定，不能只凭少GM/少dummy推断收益。
6. [上游725μs泳道](results/csa_scheduling_20260927/upstream_725/README.md)缺完整版本/Scheduler View且输入FP32，
   只作组织参考。分开end→FIN、FIN→dispatch、dispatch→start和必要多波；未计时dummy不作完整ready归因。
7. 精度版迁移、CANN9.2/新B24真实EP16 token/DSpark及10步forward仍未完成且后置。
   本页累计七档不是early单项效果，也不是EP16性能证明。

[有效策略与失败记录](DSV4_FLASH_CSA_VALIDATION_LOG.md)、[AscendC源码参考](DSV4_FLASH_CSA_ASCENDC_REFERENCES.md)。
