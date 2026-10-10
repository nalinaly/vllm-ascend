# 独立 HC_pre：AscendC profiling 与 PTO L1/L4 泳道

共 7 档 batch_size（2、4、8、16、20、24、28）× seqlen（5、6），14 个场景、42 份 trace。
每份 trace 都包含 3 次重放。T=batch_size×seqlen；两个 T120 场景分别保留。

## 打开方式

根目录 JSON 可直接在 Perfetto 打开。按文件名前缀匹配同一个场景：

- `HCpre_Bxx_Sx_Txxx_AscendC_PyTorch_3steps.json`：AscendC HcPre 的 PyTorch profiling，来自上一轮性能对比中的 Native 采样。
- `HCpre_Bxx_Sx_Txxx_PTO_L1_Swimlane_3steps.json`：同轮 PTO L1，主要查看 Worker View 中的 AIC/AIV 任务。
- `HCpre_Bxx_Sx_Txxx_PTO_L4_Swimlane_3steps.json`：同一份算子快照新采集的 L4，增加派发/完成、Scheduler View 和 AICPU Orchestrator 记录。

`index.csv` 按场景列出三个文件名及各自中位数；`manifest.json` 保留来源路径、任务编号、三个 step 的统计及版本。
`source_reports/` 是四份原始运行报告，`sources/` 是实际采集的两份独立算子源码。

## 来源及计时口径

AscendC 使用 cann-recipes-infer 的 `ops/ascendc/src/hc_pre/op_kernel`，源码版本为
`4c3d1e1258053c4d810322198a7fd1722c89897e`，调用 `custom::npu_hc_pre → aclnnHcPre`，三输出范围。
输入使用正式模型第 2 层 HC attention 权重与合成 BF16 残差，保持原固定容差及 20 次 Sinkhorn 迭代。

本包对应上一次对比表：B2/B4/B8/B16/B20 使用 v3 快照，B24/B28 使用 v4 快照。
L1/AscendC 直接复制已有采样，L4 按对应版本、相同权重和场景种子补采，分别通过精度及图重放检查。
这不是同一个时间窗口中切换显示层级；L1 和 L4 来自独立执行，不能叠加它们的绝对时间戳。
L4 采集的额外记录可能影响耗时，性能对比继续使用 AscendC 与 L1，L4 用于看调度细节。

PTO incore 首尾取全部 AIC/AIV 的最早 kernel 开始至最晚结束：包含内部转换、同步和等待，
不包含首任务开始前及末任务结束后的 AICPU 区间。Worker 条形含 local_setup，不能直接把条形总长当作 incore 时间。
报告通过 raw 记录确认真实采集档位为 1/4，并检查每份恰好三个 epoch、每个 epoch 72 个核任务；
L4 四种视图都含实际事件。没有把 L1 数据重新标记成 L4。

## 场景索引

| BS | seqlen | T | 算子快照 | AscendC μs | PTO L1 μs | PTO L4 μs（诊断） |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 2 | 5 | 10 | v3 | 34.68 | 27.64 | 32.00 |
| 2 | 6 | 12 | v3 | 35.74 | 28.34 | 31.76 |
| 4 | 5 | 20 | v3 | 38.56 | 29.72 | 31.84 |
| 4 | 6 | 24 | v3 | 40.60 | 29.18 | 32.66 |
| 8 | 5 | 40 | v3 | 48.96 | 31.70 | 33.34 |
| 8 | 6 | 48 | v3 | 50.12 | 31.88 | 33.00 |
| 16 | 5 | 80 | v3 | 55.78 | 33.70 | 34.40 |
| 16 | 6 | 96 | v3 | 60.16 | 33.88 | 35.68 |
| 20 | 5 | 100 | v3 | 61.36 | 35.40 | 35.98 |
| 20 | 6 | 120 | v3 | 66.06 | 35.22 | 36.66 |
| 24 | 5 | 120 | v4 | 61.84 | 35.48 | 33.78 |
| 24 | 6 | 144 | v4 | 62.76 | 39.32 | 36.12 |
| 28 | 5 | 140 | v4 | 62.60 | 38.80 | 35.90 |
| 28 | 6 | 168 | v4 | 69.88 | 39.96 | 38.82 |
