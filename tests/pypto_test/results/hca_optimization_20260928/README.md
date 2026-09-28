> **调测信息的权威记录已迁至 [DSV4_FLASH_HCA_OPTIMIZATION_LOG.md](../../DSV4_FLASH_HCA_OPTIMIZATION_LOG.md)。**
> 本文件保留为该目录的数据索引与当时的原始记录；结论与口径以上述 LOG 为准。

# HCA 性能优化记录

## 目标与顺序

2026-09-28 用户要求开始性能优化，目标七档全面超越 Native 20%。本轮采用更严格的耗时口径：
每档 PTO P50 ≤ 同卡 Native P50 × 0.80，保留 P95、最大值和全部原始样本，不剔除异常点。
测量完整 attention 半层：mHC pre、输入 norm、HCA、O projection、mHC post，包含 compact metadata。
沿用正式第 3 层 W8A8 权重、S6、NZ2/det0/atomic0、相同随机种子与 Native 初态。

遵循用户追加要求：**先单卡，再整机**。每版先验证单卡功能和性能；有收益后补齐七档，
单卡通过才提交优化版 TP1×DP/EP16。整机最终仍要求输出 token 一致，并核对所有目标层实际执行 PTO。
已经运行的 `task_20260928_162843_15654084376` 是优化前的精度基线，允许其正常收尾。

## 基线与首项候选

- `source_baseline/`：本地冻结的 `vllm_ascend/ops/pypto` 源码，已包含无效压缩 KV 读取修正。
- `source_page_gather/`：仅将 raw KV 的逐行搬运改为 Native 页内连续搬运。
  128+5 行窗口通常从 133 次复制缩为 5～6 次；首尾页用动态有效行，保留原来的零填充和非法页处理。
  不改计算顺序，不修改 PyPTO/Simpler/pypto-lib，不新增外部适配。
- CPU 完整设备编译：`page_gather_build/report.json`，PASS。
- 正式计时每侧预热 10 次、50 个样本，同一进程内 Native/PTO 交替；另采 PyTorch profiler。
  泳道在独立进程采两个图重放窗口，不计入正式计时。
- 所有源码快照、二进制和 states.pt 本地保留，不做 hash 校验。

| 内容 | 任务 |
| --- | --- |
| 修正后基线 8K/B16 | `task_20260928_172700_76551110341` |
| 修正后基线 128K/B16 | `task_20260928_172700_7646117717` |
| 页内搬运 8K/B16 | `task_20260928_173054_12445805436` |
| 页内搬运 128K/B16 | `task_20260928_173054_12446829144` |
| Q 展开 M128 8K/B16 | `task_20260928_173639_172681418470` |
| Q 展开 M128 128K/B16 | `task_20260928_173639_172685418840` |
| Q 展开 M128 同卡 ABBA 8K/B16 | `task_20260928_173950_188649126854` |
| PV N128 双 Acc 8K/B16 | `task_20260928_174549_211794223940` |
| PV N128 双 Acc 128K/B16 | `task_20260928_174549_211655319881` |

排队不等同于通过；性能结果以各目录 `timing/report.json` 为准。
新旧纯搬运版还需比较 `states.pt` 中的 PTO 输出与三份完整 allocation；不能用 Native 零容差代替 token 验收。

入口：`../../run_hca_performance_case.sh`；冻结源码由测试专用 `--operator-source` 选择。
泳道汇总：`../../summarize_hca_swimlane.py`。图内任务跨度有重叠，不把核内时间加和当作完整延迟。

## 已测数据与取舍

上述单卡任务均完成。页内搬运、Q 展开 M128、PV N128 三项候选的长短 B16 输出、SWA、
compressed、state 与修正后基线逐 bit 一致，保护区、只读 metadata 和图重放检查通过。
这不是七档通过，也不能代替优化版整模型 token 验收。

下表每行的 Native/PTO 为同卡同进程 50 个无 profiler 样本；不同候选之间物理卡不同，
不能只比较两个 PTO 数字就宣称同卡收益。

| 版本 | 档位 | Native P50 μs | PTO P50 μs | PTO 耗时变化 |
| --- | --- | ---: | ---: | ---: |
| 修正后基线 | 8K/B16 | 551.08 | 612.87 | +11.21% |
| 页内搬运 | 8K/B16 | 565.13 | 589.22 | +4.26% |
| Q 展开 M128 | 8K/B16 | 563.75 | 613.55 | +8.83% |
| PV N128 双 Acc | 8K/B16 | 557.41 | 596.55 | +7.02% |
| 修正后基线 | 128K/B16 | 693.72 | 712.19 | +2.66% |
| 页内搬运 | 128K/B16 | 712.07 | 694.56 | −2.46% |
| Q 展开 M128 | 128K/B16 | 694.24 | 694.29 | +0.01% |
| PV N128 双 Acc | 128K/B16 | 694.18 | 702.96 | +1.26% |

- 页内搬运的 raw gather 核内均值约 55～65 μs 降到 7～8 μs，但原任务与 Q 投影重叠，
  不能把该差值直接当作整层收益。`page_gather_h*/timing/baseline_comparison.json` 保存精确状态对照。
- Q 展开 M128 的跨卡首轮结论不稳定，补做同卡 A→B→B→A，各进程 100 个样本。
  合并两轮 P50 为 602.47→585.63 μs（−2.80%），Native 581.18→577.91 μs（−0.56%）；
  P95 为 627.20→609.72 μs。只证明该短 B16 有小幅收益，未据此保留全部档位或宣称达到目标。
  完整样本与精确对照在 `qb128_abba_h8192_b16/`。
- PV N128 借鉴 Native/CSA 的双 Acc 写回重叠，压缩 attention 的核内时间下降，但整层收益尚未确认。
  未用两个 DFX 窗口代替正式计时。其源码快照为 `source_pv128_pair/`。
- 当前在试 O projection 反量化与 mHC post 合并。公共 O 投影只抽取一次主体供两种收尾复用，
  不复制整套投影，不改分组量化、BF16 舍入点和 HC 输入累加顺序。编译/设备结果另记，不预记 PASS。

当前**未达到七档全面领先 Native 20%**。未提交优化版整机任务。

## O 反量化与 mHC post 合并

首版 `source_opost_fused` 的权重列提取不正确，长短 B16 与 H0 在输出精确对照时失败，
没有进入计时。transpose→slice→reshape 的组合改成显式 gather 到独立连续向量后，
`source_opost_fused_v2` 的 B16/H8192、B16/H131072、B1/H0、B1/H124 均通过旧 PTO
输出与全部 cache/state 精确对照、只读/写保护及服务图重放检查。H0/H124 还毒化了不可见压缩 KV。
这仅证明本次合并保持原有数值，不能替代整模型输出 token 验收。

| 内容 | 任务 | 结果 |
| --- | --- | --- |
| 首版短 B16 | `task_20260928_175916_259262415131` | 输出检查失败，未计时 |
| 首版长 B16 | `task_20260928_175916_25926672207` | 输出检查失败，未计时 |
| 首版边界 | `task_20260928_175916_259210112049` | H0 输出检查失败，H124 未执行 |
| 修正短 B16 | `task_20260928_180423_275270210927` | 功能通过；Native/PTO P50 592.12/621.07 μs |
| 修正长 B16 | `task_20260928_180423_275274228768` | 功能通过；Native/PTO P50 689.63/695.34 μs |
| 修正边界 | `task_20260928_180423_275199317320` | H0/H124 完成，exit=0 |
| 16 行收尾短 B16 ABBA | `task_20260928_181239_311254711387` | 完成，exit=0 |
| 16 行收尾长 B16 ABBA | `task_20260928_181241_311326324004` | 完成，exit=0 |

合并任务最初每块处理 32 行，B16/S6 只分派 24 个 Vector 块，核内约 40～44 μs。
`source_opost16` 改为每块 16 行，形成 48 块，算术顺序不变。
同卡 A→B→B→A 的 A 为 `source_pv128_pair`（未合并），B 为 `source_opost16`，
每个进程 100 个正式样本，各版本合并 200 个样本，全部输出与 allocation 精确一致：

| 档位 | PTO 基线→候选 P50 μs | PTO 变化 | Native 同轮变化 | PTO P95 μs |
| --- | ---: | ---: | ---: | ---: |
| 8K/B16 | 618.49→597.97 | −3.32% | −2.30% | 647.20→625.18 |
| 128K/B16 | 721.22→699.15 | −3.06% | −1.05% | 812.08→782.50 |

短档 Native 同时变快，不能把全部 3.32% 都归因于合并；目前作为小幅收益候选保留。
基线短档有 14.56 ms 的 Native 最大值，原始样本没有剔除；P50 和 P95 按完整样本计算。
原始数据与合并结果位于 `opost16_abba_h*/summary.json`。

## 下一项候选：有界压缩历史统一 softmax

长档现有压缩分支每个 128 行 K 块都在 Cube/Vector 间交接概率与 FP32 PV，
再在 Vector 累加输出。候选在压缩页表不超过 9 个 K 块时先计算完整 QK，
一次 softmax 后在 Cube 内累加完整 PV；更长历史保留原实现；本轮短历史也进入统一路径，单卡对照另要求与旧 PTO 精确一致。
每个 query 仍按 device positions/seq_lens 限制有效压缩行，未初始化 KV 的读取保护保留。
此项改变 softmax 分块及舍入，不能要求对旧 PTO 逐 bit 相同，也不能以有限值检查代替 token 验收。
先做 CPU 编译和单卡数值/性能诊断，当前未宣布通过或收益。


统一 softmax CPU 完整编译 `bulk_softmax_build_v8/report.json` PASS。
编译中遇到 HBG 条件 And 生成不支持、INT64 循环界和条件分支 Tensor 版本越出 scope 问题；
在本仓库改为单一上界条件、显式 INDEX、压缩分支使用调用者 scope 后通过，未修改依赖组件。
候选源码冻结于 `source_bulk_softmax`。概率尾部显式 fillpad(0)，保护 PV 末个 K 块。

单卡任务已提交：128K/B16 `task_20260928_182814_29029731098`，
8K/B16 `task_20260928_182815_290384016635`。短档额外要求对 `source_opost16` 精确一致；
长档改变 softmax 分块，仅先报告 Native 误差和功能检查，不预记精度验收通过。
两个任务都只占一张卡，包含独立的无 profiler 计时与 DFX 窗口。


统一 softmax 首轮两个单卡任务均 completed/exit=0。8K 输出、cache/state 对旧 PTO 全部精确一致；
128K 对 Native 的 max_abs=0.03125，RMSE 0.00129920→0.00129780，未发生明显精度恶化，
但对旧 PTO 有 115929/1572864 个 BF16 值不同，max_abs=0.015625，不能宣称 token 已验收。
128K 的三份完整 cache/state 和 Native 基线结果均逐 bit 一致，详见 `timing/baseline_comparison.json`。

首轮 Native/PTO P50：8K 586.75/628.23 μs；128K 697.32/710.89 μs。
短档压缩核内约 39～46 μs，慢于原单块流水约 22～28 μs，因此恢复单块原路径。
第二版 `source_bulk_fullpv` 将 PV 的 128 列分片改为 512 列整体累加，减少跨步列搬运，
CPU 完整编译 `bulk_fullpv_build/report.json` PASS；单卡 `task_20260928_183308_305338728094`
completed/exit=0，输出与第一版统一 softmax 精确一致，Native/PTO P50 709.65/714.95 μs。
本轮仍无足够性能收益，停止扩测，生产树已恢复 `source_opost16` 的压缩 attention。
两份候选源码及结果本地保留，没有提交优化版整机。

前段门控依赖复核：泳道中的 mix_x_rms_norm 未直接依赖 comb_sinkhorn；末尾额外依赖是
x_mixed 的分配任务。未因时间上接近就移除必要依赖，也未改公共 hc_pre 调度。


### 追加证据：整组启动仅作为独立单卡调度候选

进一步逐块统计 `bulk_fullpv_h131072_b16/swimlane/dfx`：两个窗口均为 24 个 AIC 块，
仅 16 个不同物理核；2、5、8、11、14、17、20、23 各执行两块。
启动跨度分别 110.40/107.22 μs，明显大于单块核内平均 103.05/97.88 μs。
因此暂停对统一 softmax 算法本身作性能结论，增加一个只修改 `sync_start=True` 的独立候选，
验证是否能排除这段调度拖尾。生产树仍保持 `source_opost16`；候选仅放在
`source_bulk_fullpv_sync` 本地快照，未修改 Simpler 或调度器。
CPU 编译入口增加测试专用 `--operator-source`，可以直接编译冻结源码，避免切换生产树。


调度候选任务：

| 内容 | 任务 | 状态说明 |
| --- | --- | --- |
| 压缩整组启动首提 | `task_20260928_183951_33014823469` | 多行命令被队列解析为 set 参数，exit=2；未执行 NPU 用例 |
| 压缩整组启动同卡 ABBA | `task_20260928_184221_341379725603` | 物理卡 9，独立 DFX 在四轮正式计时之后 |
| Q_B 整组启动短档首提 | `task_20260928_184222_341513030927` | 卡 11 在 SetDevice 阶段 507033/HDC 子进程启动超时，未进入算子 |
| Q_B 整组启动长档 | `task_20260928_184223_341646014912` | 物理卡 13，四轮计时已完成，等待 DFX 收尾 |

Q_B 候选仅在 `source_qb_sync` 冻结目录给 NZ 的 24 个 AIC 块加 sync_start，
基线为 `source_opost16`，计算未变。CSA 历史曾否定同一标志，但 HCA 没有 Indexer 竞争，
当前 Q_B 又在关键链上，因此只按 HCA 同卡数据判断；未改生产公共 Q projection。
短档故障用例已转投运行正常的卡 13，等待长档释放，结果使用新的 `_v2` 目录，不覆盖故障报告。


### 整组启动对照结论

Q_B 短档重提 `task_20260928_184733_375564532128` 在卡 13 完成 exit=0，
目录 `qb_sync_abba_h8192_b16_v2`；长档 `task_20260928_184223_341646014912` 也完成。
四轮各 100 个样本，后续三轮与首轮基线的输出、全部 cache/state 精确相同。

| 档位 | PTO 基线→候选 P50 μs | PTO 变化 | Native 同轮变化 |
| --- | ---: | ---: | ---: |
| 8K/B16 | 601.15→607.22 | +1.01% | +2.76% |
| 128K/B16 | 703.76→700.08 | −0.52% | +3.57% |

Q_B 整组启动没有稳定的整层收益，不合入生产实现。完整样本与 P95 在各目录 summary.json。
压缩 attention 整组启动任务 `task_20260928_184221_341379725603` 完成 exit=1：
第四轮基线在卡 9 SetDevice 阶段发生 507033/HDC 子进程启动超时，未运行算子。
前三轮 PTO P50 依次为 684.87、734.68、704.49 μs；ABBA 和 DFX 均不完整，不汇总为通过。
不重启卡、不改共享调度。生产压缩 attention 继续保持 `source_opost16`。

下一候选 `source_short_fused` 只放在测试快照：合并短历史 raw/压缩 attention 与
归并、逆 RoPE/分组写回，保留两路 softmax 的独立统计和原始 BF16 概率舍入。
先验证短档和边界，不预记收益或通过；长历史继续原实现。


### 19:10：两个独立融合候选，仅排单卡

- `source_short_fused`：短历史的 raw QK/PV、压缩 QK/PV、两路状态归并、逆 RoPE/分组
  写回合入一组 24 个 MIX 块。两路仍独立 softmax、K128 PV、BF16 概率舍入；
  双路 PV 中间结果只用按 worker 分配的固定中转空间，不再存全部 token 的六份状态。
  当前仅在压缩页表容量不超过四页时进入，宽页表继续原实现，不在 Host 读取 seq_lens。
  `short_fused_build_v3/report.json` 完整 CPU 编译 PASS，编排代码中两分支汇合后都执行 O projection。
  单卡 B16/H8192 `task_20260928_190150_994132305` 已提交，结果目录 `short_fused_h8192_b16`。
- `source_hcpre_fused`：保留原 FP32 输入与 split 0→3 的归约顺序，让两个门控消费者
  各自归约很小的线性部分和；pre/post 门控并入 mix_x_rms_norm，comb Sinkhorn 仍独立。
  编译修正均在本仓库测试源码：显式 Tensor/Tile 读写、四列偏移用 gather 避免 32B 对齐问题。
  没有改 PyPTO/PTOAS/PTO-ISA。`hcpre_fused_build_v6/report.json` 完整编译 PASS，
  已核对没有独立的 hc_pre_linear_reduce、split_pre_post 设备函数。
  这版基于 source_opost16，未叠加 attention 候选，以便单独归因。
  单卡 B16/H8192 `task_20260928_190950_35744614087` 已提交，结果目录 `hcpre_fused_h8192_b16`。

两项都先精确对照旧 PTO 输出与全部 cache/state，通过后才进入正式计时、另进程采泳道。
检查失败时脚本终止，不将有限值检查冒充输出 token 验收。
截至记录时共享机器全部卡被其他任务占用，单卡排队中；生产实现仍为 source_opost16。
优化版没有提交整机任务；七档单卡通过后才进入整机 token/性能验证。


### 19:15：短历史双槽流水的 CPU 候选

对冻结的 `source_short_fused` 只读复核后发现，第一版每个 query 都等到 PV 归并结束才
继续，Cube/Vector 仍会串行等待。因此另建 `source_short_pipeline`，不修改已排队快照：
前看一个 query、两个固定中转槽，AIV 先发布后一 query 的概率，再归并前一 query 的 PV。
PV 重新读取已经聚合的 KV，以免两套 raw/压缩 KV 同驻 L1 超限；两路归约、BF16 舍入均未改。
`short_pipeline_build/report.json` 完整 CPU 编译 PASS。该版本尚未提交设备测试，
待第一版单卡证据到齐后再决定下一步，不将 CPU 编译记为精度或性能通过。


### 19:50：资源恢复后的单卡证据与后续候选

19:35 左右单卡资源释放，19:19 的等待检查点已过时。本轮仍未提交优化版整机。
生产实现继续保持 `source_opost16`；以下新融合都只在本地源码快照中。

| 单卡 B16/H8192 候选 | 任务 | 对旧 PTO 输出、全部 cache/state | Native/PTO P50 μs | PTO 延迟变化 |
| --- | --- | --- | ---: | ---: |
| 短历史融合 | `task_20260928_190150_994132305` | 全部精确相同 | 592.23 / 581.25 | −1.85% |
| 短历史双槽流水 | `task_20260928_194020_143291422129` | 全部精确相同 | 579.59 / 579.19 | −0.07% |
| mHC pre 融合，对齐修正后 | `task_20260928_194155_147144617816` | 全部精确相同 | 566.85 / 595.45 | +5.05% |

每轮 10 次预热、50 个计时样本，profiler/DFX 使用独立窗口。服务图和保护区检查通过。
各自 `timing/report.json` 保存所有样本、P95 和最大值；对应 `swimlane/worker_summary.json`
已生成。以上是各轮同卡 Native 对比，不拿不同卡、不同时间的候选绝对值作优化归因。
双槽流水没有显示稳定增益，暂不采用其额外复杂度。

mHC pre 第一版 `task_20260928_190950_35744614087` 在首次 PTO 执行触发 507015：
VEC 访问 UB 地址未对齐。生成的 post 门控 `TEXTRACT` 起始列为 4，FP32 偏移 16B。
已保存 `hcpre_fused_h8192_b16/task.log`；因失败后运行时无法确认设备静止、退出挂住，
仅终止本会话这一个失败作业，exit=130，没有重置设备或修改其他作业。
修正版 `source_hcpre_fused_aligned` 将 pre/post 列提取也改为连续 Tile gather，
完整 CPU 编译和上表真机精确对照通过。未改依赖仓库。

泳道显示 mHC pre 的归约与门控融合能提前 normalized 的完成，但独立整层计时没有赢过
Native。因此下一步只做一次组合的同卡 A→B→B→A 对照：
`source_opost16` 对 `source_short_hcpre`（短历史第一版融合 + 对齐后的 mHC pre），
`task_20260928_194836_16190322987`，输出 `short_hcpre_abba_h8192_b16`。
后续三轮均要求对第一轮旧 PTO 输出和完整 allocation 精确相同；不预记收益。

长历史另建 `source_unified_attention_ordered`：压缩分块仍按原归约顺序，raw 最后归并，
统一 MIX 流水并融合归一化、逆 RoPE/写回。`task_20260928_193650_130585017183`
在 H131072/B16 精确门禁失败：输出 1488544/1572864 个差异、max_abs=0.46826171875；
三份 cache/state 完全一致，未计时。生成代码中 sink 与循环 running_m 共用 UB v32，
循环更新覆盖了原 sink；不能当作可接受的舍入差异。
`source_unified_attention_sink` 改为最终 raw 合并及分母阶段重新读取设备 sink 权重，
避免别名被更新，无 Host 取值，也未修改编译器。`unified_attention_sink_build` CPU 编译 PASS；
真机 `task_20260928_194732_15942363707`，目录 `unified_sink_h131072_b16`，等待结果。
这条代码证据只用于本候选修正，不作为某依赖组件的通用缺陷结论。

另建独立 `source_opost_reuse`：mHC post 的四份残差 Tile 提前读取和转换，供四个输出 HC
重复使用，仍按输入 HC 0→3 逐项相加；避免同一片残差读取/转换四遍。正在 CPU 编译，
尚未提交设备、未合入生产。全七档领先 Native 20% 的目标仍未达到。


### 20:05：短档组合有同卡收益，边界通过，开始七档单卡

`task_20260928_194836_16190322987` completed/exit=0；四轮各 100 个样本的
`short_hcpre_abba_h8192_b16/summary.json`：PTO P50 609.01→568.80 μs，下降 6.60%；
Native 559.79→556.89 μs，下降 0.52%。后续三轮输出、三份完整 cache/state 对首轮基线
精确相同。候选仍比同轮 Native 慢约 2.14%，没有达到 20% 目标。DFX 两个窗口已导出。

`task_20260928_195824_194586430795` completed/exit=0：`short_hcpre_boundaries`
的 B1/H0、H124 均通过旧 PTO 精确对照、服务图、保护区和压缩 KV 无效行毒化检查。
该组合使用短历史第一版融合，不包含双槽流水、统一长历史或 Q_B 融合候选。
七档单卡已按同卡旧 PTO 基线→组合候选排队，每轮 10 次预热、50 个样本，无 profiler；
配置、设备、任务号和完整命令保存在 `short_hcpre_seven/tasks.json`。尚未预记七档通过。

O post 残差复用 `task_20260928_195031_167457523223` completed/exit=0，输出和状态精确相同；
Native/PTO P50 559.26/598.73 μs，PTO P95=618.82、max=888.74 μs。
泳道中 post 核内耗时未显示收益，不并入组合候选，也不扩测此方向。

长历史 sink 修正 `task_20260928_194732_15942363707` 仍 FAIL：
输出 1487950 个差异、max_abs=0.40625，三份 cache/state 相同，未计时。
因此初始 sink 别名只是已发现的一处问题，不能据此声称已定位完整原因。
诊断 `attn_debug_h131072_b16_v2` 直接返回前 32 个 head 的 attention packed 中间结果，
`task_20260928_195823_19455179218` 在候选对照中确认 1570973 个 BF16 值不同，max_abs=1.158203125；
误差已经出现在 O projection 之前。该诊断不是整层正确性或性能结果。
首提 `task_20260928_195608_191275315274` 因 runner 切换工作目录、源码使用相对路径而退出，
尚未初始化 NPU；重提全部使用绝对源码与快照路径。
下一诊断 `task_20260928_200304_201669631260`，`cmp_debug_h131072_b16`：
仅比较压缩 PV 累加结果，进一步分开压缩流水与最终归并。

Q_B 独立融合候选 `source_qb_fused` 将原 N256/K1024 INT8 Cube 与 Vector
反量化、RMSNorm、RoPE 按 head 流水连接，保持矩阵分块、FP32 两步尺度乘法与 BF16 舍入。
CPU `qb_fused_build_v2` PASS；`task_20260928_195443_183704015399` 输出门禁 FAIL：
1165434 个差异、max_abs=0.158203125，三份 cache/state 相同，未计时。
每个八行块的首行相同、后七行不同。改为连续 RoPE Tile 的
`source_qb_fused_compact` / `task_20260928_195824_194638311031` 仍产生相同差异，
所以未把 gather 视图问题当作已确认的唯一根因。正在 `q_debug_h8192_b16`
（`task_20260928_200227_20078135798`）直接比较 Q，以判断 INT8 展开和 Vector 变换。

单层测试在 reference 门禁失败时新增本地 `failed_states.pt`，保留已经同步到 CPU 的结果，
仍立即中止，不计时。新增 `run_hca_reference_pair.sh` 统一绝对路径处理，默认只比对；
第六个参数 `timing` 才开启两侧计时。大快照和诊断源码目录均本地保留，不计算 hash。
生产实现仍为 source_opost16，优化版整机未提交。


### 20:08：Q_B gather 行偏移修正与当前等待点

只读复核 `rope_prepare` 确认其输出是每行 0..63 的列索引，原 Tensor gather 按行处理；
新候选误将其直接传给展平索引语义的 Tile gather，漏加 `行号×64`，因此每八行只对第一行。
`source_qb_fused_rows` 在连续 64 列 RoPE Tile 上构造完整展平索引，
`qb_fused_rows_build/report.json` CPU 完整编译 PASS。
单卡整层精确对照和计时为 `task_20260928_200633_210859118643`，
目录 `qb_fused_rows_h8192_b16`，尚未设备验收，不能预记问题已经全部解决。
先前 Q 中间量诊断 `task_20260928_200227_20078135798` 仍 pending 时已取消，
避免同一缺陷再跑两轮诊断；其源码留存。压缩 PV 诊断与七档单卡任务仍保留。

当前 16 张卡由其他整机作业占用，本会话等待既有单卡队列，不干预其他任务。
下轮先查 `short_hcpre_seven/tasks.json` 的七个句柄、压缩 PV 诊断
`task_20260928_200304_201669631260` 和上述 Q_B 修正版；不要重复提交。
生产实现仍未引入这些待验证候选，目标尚未完成。


### 20:25：长档最大值的 UB 覆盖已找到生成代码证据

七档与 Q_B 行索引修正版仍 pending，继续只读核对长历史生成代码。
在 `unified_attention_sink_build/build/kernels/aiv/hca_unified_attention_aiv.cpp`：
`selected_m`、外层 `m_after` 均指向 UB 字节偏移 94464；后续计算 `next_l` 的临时值，
以及 `pv_left`/`pv_right` 的 32×256 FP32 载入也使用 94464。
循环尾从同一地址拷贝 `m_after` 回 `running_m`，此时保存的已经是 PV 内容，最大值被覆盖。
这解释了为何只重读 sink 仍未解决差异。结构化地址记录：`unified_loop_storage.json`。

独立候选 `source_unified_attention_long` 删除长档不需要的单块压缩分支及额外 raw 最大值分支。
长档 `running_m` 从 sink 开始、后续单调取 max，因此与 raw 合并时再取 sink 的 max 是冗余的。
保留直接 `next_m=max(m_iter,out_m)` 的原长档归约结构、两路 FP32 乘法/相加顺序，
没有修改编译器、PTO-ISA 或依赖仓库；此候选仅验证长历史，不用于短历史替换。
`unified_attention_long_build/report.json` CPU 编译 PASS：`next_m/m_after` 在 UB 0，
PV Tile 在 94336，生成代码中这处覆盖已消失；设备结果仍待验证。

`task_20260928_202310_259903926056`：单卡 H131072/B16，
`unified_long_h131072_b16`，对 `qb_sync_abba_h131072_b16/0_baseline/states.pt` 精确门禁后计时。
原压缩 PV 诊断 `task_20260928_200304_201669631260` 在 pending 时取消，
改用修正版的整层对照验证，避免再跑两轮已不必要的中间量诊断。
新增 `summarize_hca_seven.py` 对齐同卡/正式配置、完整输出与状态门禁并汇总七档 P50/P95/max，
未完成的档位明确标记 PENDING，失败不计为性能通过。当前汇总 `short_hcpre_seven/SUMMARY.md`
仍全部 pending，不能把编译检查或旧短档结果当作本轮七档通过。


### 21:00：七档与两个独立候选的设备结果

`short_hcpre_seven` 七个任务全部 completed/exit=0，同卡先旧 PTO（`source_opost16`）后组合候选，
候选对旧 PTO 输出和三份完整 cache/state 精确相同。汇总见 `short_hcpre_seven/SUMMARY.md`：

| 档位 | 旧 PTO → 候选 P50 μs | 候选同轮 Native P50 μs | 候选相对 Native |
| --- | ---: | ---: | ---: |
| 128K/B4 | 471.82 → 443.04 | 440.96 | +0.47% |
| 128K/B8 | 567.18 → 563.63 | 571.05 | −1.30% |
| 128K/B16 | 706.68 → 703.80 | 697.78 | +0.86% |
| 8K/B16 | 594.60 → 568.02 | 568.46 | −0.08% |
| 8K/B24 | 732.56 → 681.25 | 719.67 | −5.34% |
| 8K/B32 | 801.11 → 741.65 | 800.62 | −7.37% |
| 8K/B40 | 915.23 → 878.24 | 875.85 | +0.27% |

组合候选使七档与 Native 大致持平，距离每档 ≤0.80×Native 仍差约 20%～26%。

- 长档统一流水修正版 `task_20260928_202310_259903926056`（`unified_long_h131072_b16`）：
  对旧 PTO 输出、SWA、compressed、state 全部精确相同，UB 覆盖问题确认已解决；
  同轮 Native/PTO P50 689.68/673.76 μs（−2.31%），PTO P95 703.08 μs。与七档中的长 B16
  候选不同卡，不作为同卡收益结论。
- Q_B 行索引修正版 `task_20260928_200633_210859118643`（`qb_fused_rows_h8192_b16`）：
  精确对照通过，确认漏加 `行号×64` 就是此前每八行只对首行的原因；但同轮 Native/PTO P50
  563.90/595.72 μs，相对 Native 慢 5.64%，与旧 PTO 同档水平相当，无收益，不合入。

### 21:05：计时口径下权重为冷数据，泳道窗口为热数据

`measure_graph_pair` 在同一进程中交替重放 Native 与 PTO 图，两侧权重合计接近 L2 192 MiB，
PTO 执行时其权重大多已被 Native 驱逐；DFX 窗口则是 PTO 图连续重放，权重热驻留。
同一 `qb_fused_rows` 版本的 PTO AICore 大核在 profiler 中为 587.65 μs，DFX 窗口为
500.7/518.2 μs；短档组合候选 DFX 为 489.0/475.7 μs，正式 P50 为 568.02 μs。
因此现有泳道低估了权重流的代价，后续泳道增加冷 L2 窗口再做归因。

短档组合候选热窗口的关键链（DFX 窗口 0，时间以 launch 为零点）：

| 阶段 | 起止 μs | 说明 |
| --- | --- | --- |
| 启动→首任务 | 0→36 | 运行时初始化与首批派发 |
| widen / mHC pre | 36→115 | 各阶段间调度空档 7～16 μs |
| Q_A、norm/quant | 118→153 | |
| Q_B `qproj_matmul` | 162→234 | 24 块只落在 21 核，3 块第二波，拖长约 35 μs |
| Q 反量化/RoPE | 240→268 | |
| 短历史 attention | 283→349 | |
| O projection | 356→438 | wo_a 64 MB + wo_b 32 MB 流 |
| O post | 445→485 | |

Q_B 第二波的原因：AIC_2/11/20 在 Q_B 派发时仍在执行 compressor 的 `hca_kv_score_proj`，
145 μs 后空闲直到 attention；其余三块排在别的核后面。AICPU Scheduler 三个线程处理每个块完成约
1～2 μs，阶段边界普遍出现 7～16 μs 的派发空档。

### 21:25：冷 L2 归因、wo_a 改 ND 与首批候选

**口径更正：wo_a 恒为 ND。** 分支已摘入 CSA 的 `1f7bad79`（原 `664c69ce`）：mode=2 下
Native 的三维 wo_a 仍为 ND(2)，此前 HCA 按 NZ 声明，`prepare_weights` 会另存一份
每层 64 MiB 的私有 NZ 副本，既多占显存，也让 PTO 享受了 Native 同配置下没有的布局
（违反 CSA §144 的同口径要求）。HCA 自有的 `decode_hca.py`、`o_proj_hc_post.py`
已同步改为 `WO_A_WEIGHT_LAYOUT`。此后的基线为 `source_prod_woa`，之前以 NZ wo_a 测得的
七档数字只作历史参考。

**冷 L2 泳道**（新增 `--swimlane-cold-l2`，每窗口前写 512 MiB 冲刷 L2）：短档组合 8K/B16
冷窗口 604～636 μs，热窗口 476～489 μs。冷时 proj_a 每块 42.9 μs（热 14.9）、qproj 55.5
μs（热 33.1），AIC 冷读单核仅约 24 GB/s。

| 候选 | 内容 | 精确门禁 | 结论 |
| --- | --- | --- | --- |
| combo_v1 | 短档组合 + 长档统一流水 | 128K 通过 | 长档 P50 661.5 vs Native 708.4 |
| v2 | 层入口 AIV 预热 ND 权重 | 通过 | kv_proj 36→12 μs，但与 widen 抢带宽，前段推迟约 25 μs |
| v3a | Q_B N128 轮间流水 | 通过 | 冷 qproj 55→69 μs，放弃 |
| v3b | proj_b stage4 | 通过 | 无变化，放弃（与 CSA §120 O-A stage4 结论一致） |
| v3c | proj_a K128 stage4 | L0B 超限 | 未上卡 |
| prod_woa | 生产 + wo_a ND | 通过 | 8K P50 640.0（Native 562.1），冷 689～700 |
| combo_v4 | v1 + wo_a ND | 通过 | 8K 570.3/543.3，128K 676.6/700.5；冷 proj_a 每块 54.4 μs |
| combo_v5 | v4 + widen 后预热 KV/cmp + Q_A 后预热 wo_a | 通过 | 冷 proj_a 54→18 μs、O 投影 206→101 μs，但 wo_a 预热与 qproj 冷读 wq_b 叠加，qproj 58→115 μs，净收益为零；计时 8K 595.5 |

DLPack 别名（让 Vector 核读取 NZ 权重字节）按用户要求暂不做。NZ 相关优化改为先对照
CSA 验证日志中已有结论：移植 d1f170ff（widen 与 RMS 合并，v6）和 56186de9（QR 输入
与 gamma UB 复用，v7），两者在 CSA 均为逐 bit 不变的核内优化。
另新增 `--swimlane-after-native`：每窗口前先执行一次 Native，复现正式交替计时里 PTO
所见的 L2 状态，用于判断预热的放置位置。

### 22:15：incore task 与 Native 逐功能对照（v8，同卡同轮）

按用户要求先对照 incore task，再看调度。新增 `summarize_hca_incore.py`：Native 取同轮计时后的
PyTorch profiler（基本单流串行，算子时长即功能耗时），PTO 取“Native 之后”泳道；PTO 给出
功能组跨度、单块核内均值与核时间，不把跨度相加当整层时长。数据：`incore_v8_h*_b16/`，
两个窗口的明细在 `incore_compare_*.json`。128K/B16 的 Native/PTO profiler 与 PTO 泳道另复制到
工作区 `hca_h131072_b16_profiles/` 供下载。同轮 P50：8K Native/PTO 551.6/597.5，
128K 684.7/685.8 μs。

| 功能 | 8K Native | 8K PTO 跨度 | 128K Native | 128K PTO 跨度 |
| --- | ---: | ---: | ---: | ---: |
| 残差加宽 | 10.0 | 15.3 | 9.5 | 14.6～14.7 |
| mHC pre + 输入 RMSNorm | 69.8 | 58.7～64.0 | 75.4 | 52.0～55.7 |
| Q_A / QR 量化 / KV 投影 | 20.0 / 18.5 / 22.3 | 14 / 7 / 15～17 | 18.5 / 21.1 / 23.5 | 15 / 7 / 15 |
| KV norm+RoPE / SWA 写回 | 33.2 / 28.9 | 8 / 9～11 | 35.8 / 29.1 | 7～9 / 10～13 |
| **Q_B 投影** | **43.6** | **56.3～56.9** | **46.9** | **54.0～85.7** |
| Q 反量化 + head norm + RoPE | 31.2 | 29～40 | 31.0 | 39～42 |
| attention + 逆 RoPE（主体） | 92.9 | 73.0～76.9 | 210.8 | 144.3～148.9 |
| **O 投影（O_A+量化+O_B）** | **138.4** | **150.6～152.1** | **152.1** | **154.7～158.7** |
| O 反量化 + mHC post | 29.9 | 33.2～33.3 | 29.6 | 33.0～33.7 |

多数 incore task 已快于 Native；关键链上明显落后的是 Q_B（单块约 47 μs，Native 22 块一波约
44 μs）与 O_A（ND wo_a 每块 38～39 μs，64 块在 24 核上约 2.7 波，并且每个 N 块重复读取
整组激活）。PTO 整层与 Native 持平主要由关键链阶段空档（8K 约 116 μs，含启动）与
运行时单次调用固定开销构成；后者属公共优化，本会话不处理。

据此排队的候选：
- v10：Q_B 每轮 N256 内按 K256 双缓冲、M128 单块覆盖有效行（首段 `matmul` 生成 compact
  累加器）。INT32 精确；CPU 完整编译通过；长短 ABBA、候选用例与 B1 边界对照已排队。
- v11：hc_pre_linear 结束后 19 μs 的派发空档与 76 个块完成同时到来有关；将 attention 的
  `hca_raw_valid`、`hca_inverse_rope_sign` 改为依赖 SWA 写回完成（Q_B 期间执行），
  KV 预热 24→8 块。纯调度；CPU 编译通过；长短 ABBA 已排队。

## 2026-09-28 23:00 CANN 从 9.0.0 切到 9.2.0-beta.2：数据分界线与重取

另一个会话在 **21:59:13** 把公共入口 `env-dsv4-0251rc1.sh` 改成 source
`cann-9.2.0-beta.2/set_env.sh`（原先是 `/usr/local/Ascend/cann-9.0.0`）。
`run_hca_single_layer.sh` 第 10 行 source 这个入口，所以任务按**启动时刻**继承版本：

| 版本 | 属于该版本的取证 |
| --- | --- |
| CANN 9.0.0 | 七档基线、`source_prod_woa` 对照、v1–v9 全部候选、`incore_v8_h*_b16`、launch floor 探针、交付的 `hca_h131072_b16_profiles/` |
| CANN 9.2.0-beta.2 | `abba_v8_v10_*`、`abba_v8_v11_*`、`abba_v8_v12_*`、`seven_cann92_v12/`、`incore_cann92_v12_*`、`metadata_replay_cann92/` |

核对方式：结果目录里 `build_output`／日志中出现的 CANN 路径。
`incore_v8_h8192_b16/timing`（21:57 运行）里是 `/usr/local/Ascend/cann-9.0.0`，
`abba_v8_v10_h8192_b16/1_candidate`（22:17 运行）里是 `cann-9.2.0-beta.2`。

同一份 v8 代码跨版本的绝对耗时（注意含换卡漂移）：

| 档位 | 9.0 Native/PTO | 9.2 Native/PTO | PTO/Native |
| --- | --- | --- | --- |
| 8K B16 | 551.6 / 597.5 | 558.3 / 597.5 | 1.083 → 1.070 |
| 128K B16 | 684.7 / 685.8 | 703.4 / 708.4 | 1.002 → 1.007 |

两侧都略慢，比值几乎不动，**0.8×Native 的目标在 9.2 上一样远**，候选方向不变。结论影响：

- **ABBA 结论仍然有效**：baseline 与 candidate 在同一任务、同卡、同 CANN 下跑
  A→B→B→A，版本是共同项。v10（8K 相对 Native +5.87%→+2.82%，128K +0.82%→+0.56%）
  与 v11（8K +5.72%→+4.33%，128K −0.31%→−0.82%）都成立，两者保留。
- **绝对基线与 incore 对照表要在 9.2 上重取**：前者是验收分母，后者是"哪个 task 落后
  Native"的判据，而 9.2 可能改了 MatMul/TransposeBatchMatMul 实现。上表那份 22:15
  的 incore 数据是 9.0 的，Q_B 与 O_A 两个落后项需在 9.2 上复核后再继续投入。

### v10+v11 合成为 v12

v10 改 `deepseek_v4_flash_dspark/q_projection.py` 与
`deepseek_v4_flash_dspark_perf/qkv_proj_rope.py`；v11 改
`deepseek_v4_flash_hca/decode_sparse_attn_hca.py` 与 `weight_warm.py`。
文件集合不相交，`source_combo_v12` 由 v8 叠这四个文件生成，已逐文件 `cmp` 校验来源。

### 新增 `submit_hca_seven_tier.sh`

七档（128K×B4/B8/B16、8K×B16/B24/B32/B40）一次性各占一张卡并行提交，
每档跑 `run_hca_reference_pair.sh`（基线→候选，候选对基线快照做逐 bit 门禁并计时）。
注意本机 `task-submit` 只接受**一整条命令字符串**；按 argv 传会被它当成自身选项，
任务会以 `exit=2` 立刻失败、日志里命令显示成 `bash --device N`（v12 首次提交即如此，已重投）。

## 2026-09-28 23:30 proj_a 定位：不是 cube 浪费，是 ND 列切片的跨步读

用户指出"参考 CSA 的算子，可以对不同的 shape、档位实施不同的核内 tiling 策略"。
查证：CSA 验证日志第 175 节 v3 的 O projection 自适应分块（O-A 选 N128/N256、
O-B 选 M32/M96/M128、大档位 O-B 权重完整 K 常驻、小档位 K256 流水，并且明确保留
Native 的 `[G,K,N]`／`[G*K,N]` 描述与原指针、不重排）在 HCA 侧已经是 `decode_o_proj_tp1`
的逐字拷贝，`weight_resident` 分支也在。**CSA 没覆盖的只有 proj_a 的行块 M。**

据此排了两个互斥假设的单变量候选，都从 v12 分叉：

| 候选 | 改动 | 8K 净收益 | 128K 净收益 | 结论 |
| --- | --- | ---: | ---: | --- |
| v13 | proj_a 行块按档位贴合（32/48/96/80×2/96×2/120×2），行块数不变 | +5.51 μs | +0.11 μs | **否**：cube 本就按 valid_shape 只算有效行 |
| v14 | proj_a 列块 N128→N256 下放到 ≤96 档 | +7.13 μs | −7.05 μs | 跨度无变化，但核内效率证实有效 |

净收益 = PTO 候选轮均值 − PTO 基线轮均值 − Native 的同期漂移（Native 代码四轮相同，
其变化量即该卡漂移）。v13 因此丢弃。

v14 的 incore 取证给出了真正的定位（8K/B16）：

| 版本 | proj_a 块数 | 单块均值 μs | 核时间 μs | 核时间/24 | 跨度 μs | 24 核填充率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v12（N128） | 64 | 41.0 | 2624 | 109.3 | 129.2 | 89% |
| v14（N256） | 32 | 64.0 | 2048 | 85.3 | 128.4 | 67% |
| Native | — | — | — | — | 86.2～87.4 | — |

N256 把核时间降了 22%，折算到 24 个 AIC 是 85.3 μs，**单核效率已经追平 Native 的
86～87 μs**；但块数从 64 掉到 32，填充率 89%→67%，利用率损失把带宽收益全部吃掉。
原因是 ND 的 wo_a 按列切片读取时，N128 每次只有 256 字节连续、跨步 2048 字节
（达成 64 MiB/129.2 μs ≈ 519 GB/s），N256 连续段翻倍到 512 字节（≈ 778 GB/s）。

### v15 / v16：N256 + 3 行块补回填充率

要同时拿到 N256 的读取效率和满波填充，只能靠增加行块：8 组 × 4 列块 × 3 行块 = 96 块，
24 个 AIC 正好 4 个满波。代价是 wo_a 被重复读 3 遍，而它当前是 `CachePolicy.BYPASS`，
3 遍就是 192 MiB HBM。因此 v15 同时把 wo_a 的缓存策略改回默认（64 MiB < 192 MiB L2），
并把块索引从"行块最外"改成"列块最外"，让同一权重列块的各行块编号相邻、落在同一调度波，
只有第一个行块从 HBM 取权重。v16 是同样分块但保留 BYPASS，用来隔离缓存策略这一项。

这一步与 CSA v2 记录的"O-A 权重 L2 bypass 有收益"方向相反，但那条结论是在单行块、
无重复读的前提下成立的；出现重复读之后 bypass 不再正确。

proj_b 的 ROW_TILE 在两个候选里都逐档保持 CSA 取值（≤32→32、≤96→96、其余 128），
已逐档校验，保证只有 proj_a 在变。两版 CPU 完整编译 PASS，长短 ABBA 与 v15 的 incore 已排队。

### 工具的系统性偏差（重要）

`run_hca_reference_pair.sh`（七档用它）是先基线、后候选顺序跑两个进程，候选总在更热的
状态下测，所以"基线→候选"的差值**系统性偏向候选**，不能当收益。两次七档之间也不可比：
同一份 Native 代码在 8K/B16 上一次 583.9、一次 544.8，差 39 μs。七档里可信的只有
**同一轮内**测出的 PTO/Native 比值。v12 轮均值 1.008、v13 轮均值 1.010，与 ABBA 的
"v13 无效果"一致。只有 ABBA 的 A→B→B→A 排列能抵消单向漂移。

### metadata A→B→A 同地址重放 PASS

`metadata_replay_cann92/`：B4、history A=124 / B=8190、统一页表宽度、B 的页表行反序。
54 个 metadata 张量叶子，其中 23 个在 A/B 间指针与内容都不同（block_table、
input_positions、qli_metadata、sas_metadata、seq_lens、start_pos 等）。A→B→A 三步
每步都是图重放与 eager 逐 bit 相同、写保护按目标状态自身 slot 计算且 PASS、输出无非有限值；
B 另与在自身张量上的直接调用逐 bit 相同，第二次 A 与第一次 A 逐 bit 相同。
对应交接文档剩余事项第 3 项。修过一个脚本缺陷：遍历 metadata 时对类对象取
`vars()` 会拿到 `__abstractmethods__` 而抛 `AttributeError`，已跳过类/模块/函数并给
`getattr` 加保护。

## 2026-09-28 23:45 v17：wo_a 布局跟随 Native（9.2 上是 NZ），前两轮 proj_a 结论作废

用户要求**性能测试优先 vllm_ascend 的 nz_mode=2**。查证后发现一个使前面两轮结论失效的前提：

我的候选快照连带冻结了共享的 `deepseek_v4_flash_dspark/nz_mode.py`，里面是
`WO_A_WEIGHT_LAYOUT = None`（写死 ND）。而 `wo_a` 在 Native 侧是 ND 还是 NZ 取决于当前
CANN 的 `npu_format_cast` 支不支持三维 BF16：**9.0.0 不支持（Native 留 ND），9.2.0 支持
（Native 是 NZ）**。所以在 9.2 上，写死 ND 会让 `native_adapter.root_weight`
`npu_format_cast` 出一份私有 ND 副本，每层 78.18 MiB、21 层合计 1.603 GiB
（数据见 CSA 侧 `results/mem_128k_b24_20260928/ANALYSIS.md` 第 19 节）。

**结论：我在 9.2 上量的 proj_a 一直是那份 recast 副本，不是 nz_mode=2 的生产路径。**
"ND 列切片跨步读只有 519 GB/s"这个定位对 recast 副本成立，但生产路径上 NZ 的列切片本身
就是连续的。因此 v13/v14/v15/v16 四个候选全部作废并丢弃：

| 候选 | 改动 | 8K 净收益 | 128K 净收益 | 结论 |
| --- | --- | ---: | ---: | --- |
| v13 | proj_a 行块按档位贴合 | +5.51 | +0.11 | 否 |
| v14 | proj_a 列块 N128→N256 | +7.13 | −7.05 | 否（核内效率有效但填充率抵消） |
| v15 | N256 + 3 行块 + 列块最外 + wo_a 走 L2 | +44.89 | +44.56 | 否，明显更差 |
| v16 | 同 v15 但保留 BYPASS | +52.50 | +39.28 | 否，明显更差 |

v15 的 proj_a 核时间 3398 μs（96 块 × 35.4），比 v12 的 2624 还差——3 个行块把权重读取
翻三倍，L2 没接住。

### v17 = v12 + 生产版共享文件

只刷新那 3 个提交（1f7bad79 / 715e504d / 1a88c7b4）动过的 `ops/pypto` 文件：
`dspark/{nz_mode,native_adapter,decode_o_proj,decode_csa}.py`、
`dspark_perf/{nz_mode,decode_csa}.py`、`variant.py`。唯一实质变量是 `WO_A_WEIGHT_LAYOUT`
改为由 `_native_keeps_3d_bf16_as_nz()` 探测，9.2 上自动变 NZ。CPU 编译 PASS。

ABBA 净收益（扣掉 Native 同期漂移）：

| 档位 | PTO v12 | PTO v17 | Native 漂移 | v17 净收益 |
| --- | ---: | ---: | ---: | ---: |
| 8K B16 | 564.48 | 534.02 | −23.48 | **−6.98 μs** |
| 128K B16 | 677.04 | 643.02 | +7.05 | **−41.07 μs** |
| 128K B24（新档位） | 831.68 | 813.05 | −3.38 | **−15.26 μs** |

proj_a 的 incore 变化（均为 64 块）：

| 版本 | 单块均值 μs | 核时间 μs | 核时间/24 | 跨度 μs | Native 同功能 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v12（recast 出的 ND）8K | 41.0 | 2624 | 109.3 | 129.2 | 86.2 |
| **v17（NZ）8K** | **22.0** | **1408** | **58.7** | **75.3** | 70.1 |
| **v17（NZ）128K** | **22.2** | **1421** | **59.2** | **80.4** | 82.8 |

核时间降 46%；整个 O 投影组从落后 +23.5 μs 变为 8K −1.9、**128K −18.0（领先 Native）**。
整层 DFX 跨度 8K 从 563.1 降到 498.6 μs，128K 为 629.0 μs。
`PTO_CSA_WEIGHT_RECAST` 告警在 v17 的所有运行里都是 **0**。

### 八档验收表（CANN 9.2，v17 对 `source_prod_woa`，逐 bit 门禁全 PASS）

| 档位 | Native P50 | 基线 PTO | v17 PTO | v17/Native | 目标 0.8× | 还差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128K B4 | 406.5 | 472.7 | 403.5 | 0.993 | 325.2 | +78.3 |
| 128K B8 | 541.8 | 581.6 | 503.9 | **0.930** | 433.4 | +70.5 |
| 128K B16 | 692.7 | 750.1 | 654.9 | 0.946 | 554.1 | +100.8 |
| 128K B24（新） | 859.1 | 887.0 | 809.8 | 0.943 | 687.3 | +122.5 |
| 8K B16 | 538.1 | 636.7 | 528.7 | 0.983 | 430.5 | +98.3 |
| 8K B24 | 680.1 | 764.9 | 653.6 | 0.961 | 544.1 | +109.5 |
| 8K B32 | 761.9 | 822.4 | 735.4 | 0.965 | 609.6 | +125.8 |
| 8K B40 | 823.4 | 945.7 | 823.2 | 1.000 | 658.7 | +164.5 |

比值均值 0.965（v12 时是 1.008），八档全部 ≤1.0，距 0.80× 仍差 70～165 μs。
**128K/B24 单卡档位已跑通并纳入验收表。**
