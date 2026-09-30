# HCA O-B激活复用

基底`bef9f7fa`，只提取CSA提交`5c66a59b`的`_proj_b_mm_nz_kernel`函数。
参考ops-nn的AL1-full/N-first：ROW≤96时，同一份INT8激活载入L1后供两个N256输出复用。
权重仍按K256双缓冲，生成码仍为K128 L0、Right 32KiB双槽；ROW128与ND路径不改。
分组量化、INT32累加顺序、任务数、依赖和HCA收尾均不改。

## 整段数据

CANN9.2、NZ2、atomic0、det0、正式第3层权重、合成历史、服务入口npugraph_ex。
同卡ABBA每侧18次纯设备span，单位μs；最大/最小值来自全部单次重放。

| 档位 | base min/max/mean | al1 min/max/mean | mean变化 |
| --- | --- | --- | ---: |
| 128K/B16 | 543.00 / 627.50 / 571.54 | 556.25 / 629.50 / 573.69 | +0.38% |
| 8K/B16 | 423.25 / 491.50 / 439.15 | 426.25 / 508.25 / 446.24 | +1.61% |

不宣称整段提速；两档最大值也没有改善。同进程交错筛选另见
[测量记录](../hca_pair_screen_20260930/README.md)，也未证明整段收益。

## 核内与联动

另采各一张独立DFX；以下每侧64个O-B worker，单位μs。

| 档位 | base核内min/max/mean | al1核内min/max/mean | mean变化 | 组跨度base→al1 |
| --- | --- | --- | ---: | --- |
| 128K/B16 | 7.80 / 17.36 / 11.83 | 7.50 / 11.72 / 9.54 | −19.37% | 67.48→60.94 |
| 8K/B16 | 6.76 / 11.78 / 8.36 | 6.42 / 9.04 / 7.79 | −6.76% | 48.98→50.14 |

长档O-A未改但mean也从27.56变为21.60；短档O-A为15.23→15.53。
因此DFX不是孤立核的因果测量，不能把所有联动变化都归因于激活复用。
短档O-B核内下降而组跨度增加，说明局部收益还没有转化为调度收益。
按“核内收益保留、后续继续优化调度”的要求接入该函数，后续基线必须包含它；
这项取舍不等于Native或整模型性能验收通过。

两档输出、完整三组cache/state逐bit一致，eager/graph与保护区检查通过。
另用128K/B4验证ROW32和24行有效输入的尾部，完整状态与保护区也通过；
该边界运行只有一个ABBA周期，仅作正确性检查，不列为性能结论。

## 证据与复现

- 长档任务`task_20260930_015816_87349716364`，短档`task_20260930_020441_100650215152`。
- B4尾块任务`task_20260930_021238_114383631021`。
- [长档结果](result_h131072_b16.json)、[短档结果](result_h8192_b16.json)、[B4检查](boundary_h131072_b4.json)。
- 每份结果保存ABBA原始span、精度状态和DFX路径；大文件在`results/hca_ob_activation_20260930/`。
- `prepare.py`冻结整包并生成[源码差异](al1.patch)，两个CPU编译报告随目录保留。
- `run.sh HISTORY BATCH`必须通过`task-submit --device auto`运行。

汇总命令为`hca_residual_reuse_20260930/collect.py --experiment-root hca_ob_activation_20260930
--candidate-label al1 --history HISTORY --batch BATCH`，在公共环境下用CPU执行。
