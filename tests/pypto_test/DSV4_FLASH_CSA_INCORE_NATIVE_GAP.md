# CSA：最新七档核内差距与下一步

更新：2026-09-29。完整七档对应 **632dd00a / CANN9.2**；包含O-B激活L1复用、整行HC收尾融合、七处early及Q/Sparse整块Gather。
七档统一冻结源码：128K B4/B8/B16/B24、8K B16/B24/B32。两代表档复用，其余五档补齐；
Native沿用已完成最新标准；Native/补五档为auto设备0，复用两代表档为auto设备1，采样分属不同任务。
这是现有结果对照，不能归因单项收益。

Native整体为npugraph_ex、dynamic=False/fullgraph=True、inplace_pass/static/SuperKernel开启；
核内诊断仅关闭SuperKernel。PTO保留自身custom-op图。两侧mode2/det0、PTO atomic0、EPLB关闭，
真实第二个CSA层权重与合成历史，完整HC_pre+norm+CSA+HC_post，5预热20次正式设备事件。
Native仍为release custom二进制；ops-transformer28f40354/ops-nn19614968/ops-math361722c0是源码参考。

[完整结果](results/csa_flat_gather_seven_20260929/RESULTS.md)、[任务明细](results/csa_flat_gather_seven_20260929/TASKS.md)、
[七份PTO泳道下载](results/csa_flat_gather_seven_20260929/download_pto_swimlanes/README.md)。

## 完整CSA

| 档位 | Native均值μs | PTO均值μs | PTO变化 | Native/PTO P95μs |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4 | 748.298 | 642.614 | -14.123% | 751.100/652.380 |
| 128K/B8 | 859.563 | 733.228 | -14.698% | 861.620/745.140 |
| 128K/B16 | 1130.853 | 964.806 | -14.683% | 1135.580/978.360 |
| 128K/B24 | 1281.888 | 1230.986 | -3.971% | 1289.880/1247.340 |
| 8K/B16 | 757.482 | 744.598 | -1.701% | 760.120/753.860 |
| 8K/B24 | 915.371 | 909.341 | -0.659% | 920.340/924.280 |
| 8K/B32 | 1062.079 | 1038.443 | -2.225% | 1066.220/1055.960 |

各上下文内batch等权，128K -11.869%、8K -1.528%，8:2 -9.801%。
七档均值均低于Native；8K/B24 P95仍略高，140次无5%尖峰不代表历史尾部已解决。
七档图重放八类状态/保护区与28窗通过，跨版本状态依据仍为两代表档及尾行/padding。
本轮不是Native/PTO跨实现或真实模型token/DSpark及EP16 forward验收。

## Indexer核内

Native独立QLI包含系数、Score和归并，PTO拆成独立任务；PMU与worker计时边界不同。
单位μs，PTO为四窗均值。以下差值用于定位，不是严格纯算术差或可回收时长。

| 档位 | Native QLI AIC/AIV | PTO Score AIC/AIV | PTO系数 | PTO merge |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4 | 66.363/64.369 | 64.517/71.367 | 1.590 | 8.273 |
| 128K/B8 | 126.316/126.021 | 115.175/119.481 | 1.780 | 10.076 |
| 128K/B16 | 242.293/241.730 | 223.657/226.679 | 2.901 | 9.890 |
| 128K/B24 | 376.467/376.063 | 330.191/337.861 | 2.303 | 13.820 |
| 8K/B16 | 39.630/39.050 | 40.567/45.043 | 6.235 | 8.747 |
| 8K/B24 | 56.051/55.666 | 30.576/45.381 | 3.312 | 8.513 |
| 8K/B32 | 68.901/68.365 | 77.159/81.395 | 3.683 | 11.351 |

长S6完整query单根，短B16/B32为双query、短B24为S6；系数/scale提交/归并仍独立。
主性能Native含QLI→Sparse SuperKernel，不将其拆成这里的独立核时。
[算术与输入差异](DSV4_FLASH_CSA_INDEXER_NATIVE_GAP.md)。

## Sparse attention

PTO已融合最终归一化、逆RoPE和发布；Native下列Sparse不含独立逆RoPE，不能机械相减。

| 档位 | Native Sparse AIC/AIV | PTO融合Sparse AIC/AIV |
| --- | ---: | ---: |
| 128K/B4 | 53.214/57.090 | 46.034/49.832 |
| 128K/B8 | 95.667/99.998 | 78.349/81.856 |
| 128K/B16 | 167.255/169.662 | 148.476/152.117 |
| 128K/B24 | 221.309/223.867 | 220.398/223.808 |
| 8K/B16 | 95.677/98.084 | 123.951/127.339 |
| 8K/B24 | 166.763/169.371 | 175.114/178.563 |
| 8K/B32 | 210.209/212.812 | 233.531/237.101 |

长B24融合Sparse 220.398/223.808μs已接近Native独立Sparse 221.309/223.867μs；
短B16/B32仍是明显参考差距。PTO包含逆RoPE，不能把差额直接解释成纯算术损失。

## Q/O及融合收尾

128K/B16、B24四窗均值。启动分散包含资源争用与必要多波，不全是调度器软件开销。

| Task | B16 worker/核时μs/启动分散μs | B24 worker/核时μs/启动分散μs |
| --- | ---: | ---: |
| hc_widen_rms | 12/13.343/1.245 | 18/13.069/4.030 |
| hc_pre_linear | 24/7.207/8.215 | 24/9.925/5.900 |
| comb_sinkhorn | 12/15.065/0.330 | 18/15.800/3.710 |
| mix_x_rms_norm | 12/17.593/0.310 | 18/18.248/3.385 |
| qproj_matmul | 24/39.288/55.260 | 24/46.901/68.350 |
| qproj_dequant_rms_nope_rope | 48/22.394/12.885 | 48/31.368/9.535 |
| proj_a_mm | 64/27.450/67.760 | 64/27.039/66.145 |
| quant | 24/7.973/56.735 | 40/6.949/60.025 |
| proj_b_mm | 64/10.201/47.955 | 64/18.567/63.040 |
| proj_b_act_hc_post | 24/27.465/0.495 | 36/31.123/0.940 |

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
   八类完整状态、16窗及H127/B3/padding通过，仅性能版单文件。本页632dd00a七档表已包含该项。
5. 保留长小batch Score、长B24与短档Sparse的核内入口；128K优先、长短8:2，真实核内收益保留，
   明显场景分化在同一算子内部按输入选择。两版Sparse计划提前已否定，不能只凭少GM/少dummy推断收益。
6. [上游725μs泳道](results/csa_scheduling_20260927/upstream_725/README.md)缺完整版本/Scheduler View且输入FP32，
   只作组织参考。分开end→FIN、FIN→dispatch、dispatch→start和必要多波；未计时dummy不作完整ready归因。
7. 精度版迁移、CANN9.2/新B24真实EP16 token/DSpark及10步forward仍未完成且后置。
   本页累计七档不是early单项效果，也不是EP16性能证明。

近期[NZ O-A按形状扩大L1面板](results/csa_oa_l1k512_20260929/README.md)验证已结束：
参考最新ops-nn的L1 depth/step选择，仅T≤96/N128改K512，T>96/N256与ND保留K256。
L0仍K128双缓冲、任务/worker及K顺序保持；CPU完整编译通过，N256规范化IR与基线相同。
两档已完成：目标核长+6.663%、短−8.638%，四窗范围各自不重叠；8:2核内+3.603%、CSA−0.159%。
八类状态/16窗通过，不统一采用K512。按用户规则改为仅短历史T96的分支候选，长档/其他形状保持；
复用既有长度扫描后，短档分支的两档实测亦未获益：短O-A +6.175%，CSA 8:2 +0.426%、目标核+0.353%，
八类状态/16窗通过、P95两档升高。不采用该分支，生产仍K256；不同轮次卡/编排不同，不拼旧最优数字。
八张静态B的独立NZ对照及两个单项反序复核已结束：没有确认稳定缩短完整CSA的候选，生产未接入。
七项完整状态通过；Hadamard出现显著Top-K/输出差异而拒绝。核内更快不等于区间收益，
[逐项/组合/复核证据](results/csa_static_b_nz_20260929/README.md)保留全部取舍。

已采用的Sparse最终逆RoPE整块Gather参考AscendC按整块处理索引的实现。
A3二维TGATHER已有16轮向量barrier，展平1×1024后只需一轮，元素总数与全部浮点运算保持。
CPU确认无新增搬运。两档Sparse AIV加权−2.472%、完整CSA−0.266%；长CSA+0.168%、P95+6.880μs单列。
八类完整状态、H127/B3及同图padding通过，已接入性能版单文件。本页632dd00a七档表已包含该项。
[候选与源码依据](results/csa_sparse_rope_flat_gather_20260929/README.md)。

[有效策略与失败记录](DSV4_FLASH_CSA_VALIDATION_LOG.md)、[AscendC源码参考](DSV4_FLASH_CSA_ASCENDC_REFERENCES.md)。

Indexer query整块Gather已完成独立两档A/B，CPU处理了64列切片实际行距128的问题；
两档及H4095/B3真实筛选/尾行/padding八类状态精确，目标核时8:2−15.193%但CSA+0.240%；已按核内收益规则接入性能版。本页七档仍为修改前632dd00a，不外推本项收益。[候选](results/csa_indexer_rope_flat_gather_20260929/README.md)。
[固定window_3调度拆分与727.98μs参照](results/csa_flat_gather_seven_20260929/SCHEDULING.md)
显示短B16前段/中段仍长，Sparse含发布段更短；旧上游无Scheduler View，不计算纯AICPU差值。
