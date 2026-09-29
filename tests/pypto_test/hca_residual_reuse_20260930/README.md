# HCA mHC post residual复用

基底c6792787；性能目标仍为各档快于同配置Native 20%以上，当前是长档单卡候选筛选。
固定CANN9.2.0-beta.2、NZ mode2、atomic0、det0、BF16残差、正式第3层权重和合成历史。
本轮无Native重测，不把历史1.117比值当作本轮已复核基线。

## 改动依据

本地最新ops-transformer的`mhc/mhc_post/op_kernel/arch22/mhc_post_arch22.h`提供
`USE_PERMANENT_X`分支：`CopyInX`一次读取全部HC分量，`ComputeCopyOutAllX`一次转FP32，
随后四个输出复用。这里只借鉴实现，不声称Native当前tiling必然选中了这个分支。

HCA原实现外层out_h展开4次，内层in_h流水每次重新加载和cast4份residual。
候选把四份8×512 FP32 residual保留在UB，加载/cast次数由16降到4；
额外驻留需要检查容量，不能只按源码加载数宣称收益。
输出仍先post×attention，再按in_h=0/1/2/3逐项加，保留attention落BF16及最终BF16舍入。
没有调整SPMD组数、task依赖、sync_start、其他算术或cache布局。

`prepare.py`整包冻结基线和候选，记录source.json与reuse.patch；全部Python文件只读。
`compile.py`显式解析两根依赖图，并编译、加载完整设备代码，两侧CPU通过。
这证明“仅import/register不触发trace”不等于“CPU不能编译校验”；CPU结果不替代设备正确性。

## 测量范围

task_20260930_011026_17930231765，自动队列单卡128K/B16。
`run.sh`固定快照按ABBA四个进程执行，各20次图外事件、9次独立profiler重放；
再分别采基线和候选独立泳道。结果目录：
`../results/hca_residual_reuse_20260930/h131072_b16/`。

`collect.py`区分全部单次设备span的min/max/mean和pass中位数分布，不拿后者的max替代真实单次max。
图外事件含主机间隔，单列作诊断，不与纯设备span混算。
完整状态在性能之后比较，只读取已有states.pt，不重复上卡做同一项检查。
自身eager/graph一致、写保护通过不代表对Native token或整模型验证通过。

## 结果与取舍

长档任务已完成；补充8K/B24任务`task_20260930_011655_2614578312`也已完成。
两档均为同卡ABBA，每侧18次profiler重放；下表统计全部单次设备span，单位μs。

| 档位 | 实现 | min | max | mean |
| --- | --- | ---: | ---: | ---: |
| 128K/B16 | base | 565.50 | 617.50 | 583.31 |
| 128K/B16 | reuse | 568.00 | 636.00 | 599.17 |
| 8K/B24 | base | 604.50 | 646.50 | 625.18 |
| 8K/B24 | reuse | 603.25 | 662.50 | 620.08 |

独立DFX窗口的post核内min/max/mean：

| 档位 | 实现 | min | max | mean | task首开到末结束 |
| --- | --- | ---: | ---: | ---: | ---: |
| 128K/B16 | base | 32.28 | 38.80 | 35.26 | 39.00 |
| 128K/B16 | reuse | 22.30 | 28.96 | 26.09 | 32.22 |
| 8K/B24 | base | 18.72 | 42.44 | 31.93 | 62.30 |
| 8K/B24 | reuse | 17.70 | 38.60 | 29.74 | 58.16 |

长档48个worker、短档72个worker，两侧任务形状相同；短档有不同row工作量，
不能将72个worker的极差直接解释成执行抖动。
长档post核内mean降低26.01%，短档降低6.85%；基于用户允许保留incore收益的要求接入生产。
整层长档mean反而增加2.72%，短档减少0.82%，两档max均增大；尚无整层性能收益结论。
长档独立泳道中未修改的attention AIC mean155.03→169.26、O-A mean24.03→29.81，
实际启动与核内耗时均有变动。不能把整层变化单独归因于post或某个未经测量的硬件因素。
后续在新基线上研究任务联动，不把本轮结果当作已达Native 20%目标。

两档各四份已保存状态与同档首份base输出及完整cache/state逐bit一致；
自身eager/graph和保护区检查通过。尚未补齐七档或重新做整模型token验收。
详细样本、路径及核内信息在`result_h131072_b16.json`、`result_h8192_b24.json`。
已有的group/swapnrw/postsplit实验终态汇总在`inherited_schedule_results.json`，
这些历史实验没有保存跨变体状态，不能外推为数值中性验证通过。
