# 四组 Q 的启动同步与生产者消费者联动（2026-09-30）

生产基线0742f07c，CANN9.2/NZ2/atomic0/det0、正式第3层。
每组4 Cube与12 AIV，合计16 Cube/48 AIV；保持核数、分块、算术、TaskId和提前解析设置。
只比较四个Cube组各加sync_start，以及Cube＋对应反量化组同时加sync_start。
旧整组Q同步的结论不能代替当前四组发布结构的实测。

## 无 DFX 的完整区间

同进程ABBA每侧10个真实设备span，μs，不同轮只与自己的base比较。

| 档位/候选 | base min/max/mean | candidate min/max/mean |
| --- | --- | --- |
| 128K/B16，仅Cube同步 | 535.000/584.750/547.600 | 546.000/571.000/559.950 |
| 128K/B16，Cube＋反量化同步 | 566.250/602.500/576.275 | 567.000/587.750/578.525 |
| 8K/B24，Cube＋反量化同步 | 592.500/636.250/604.425 | 588.500/621.000/606.700 |

仅Cube同步mean增加12.350 μs，五个ABBA小组全部回退，不保留为生产设置。
联动同步长短mean分别增加2.250/2.275 μs，8:2相对加权约+0.388%；P50也分别增加6.125/2.375。
样本max分别下降14.75/15.25，但两档base的最大值均是整个profile的第一个重放。
短档后续base的max为609.25，低于候选621.00；不能把首个样本主导的max差称为稳定削峰。
所有样本照常报告，不事后剔除第一项来改写正式min/max/mean；只记录这一证据限制。

两候选长档、联动短档完整输出/cache/state跨版本逐bit、自身eager/graph与保护区通过。
没有必要扩展无完整区间收益的七档、尾块或整机测试。生产保持0742f07c的算子。

## 独立泳道

另一次同卡DFX比较base与联动同步，μs；不是上述ABBA的同一窗口。

| 项目 | base | 联动同步 |
| --- | --- | --- |
| 四组Cube跨度 | 92.88/102.54/82.12/75.98 | 80.92/76.54/77.82/74.76 |
| 四组反量化跨度 | 43.02/65.78/40.40/42.60 | 37.32/37.20/37.32/38.14 |
| 最后一组反量化完成 | 335.98 | 314.20 |
| 并发压缩gather跨度 | 36.56 | 31.66 |
| Attention AIV启动 | 337.36 | 316.42 |

这个窗口确认组内起跑更整齐、相关消费者更早就绪；但不能把21μs的局部时序变化
直接说成实际完整HCA快了21μs。无DFX的长短ABBA没有相应收益，因此同步联动只保留候选和证据。
这也不是“所有sync_start无效”的结论，下一次改核内实现后仍需按新并发关系判断。

## 复现与取舍

[source.json](source.json)、[summary.json](summary.json)、[incore.json](incore.json)、[tasks.json](tasks.json)。
源码差异为[cube_sync.patch](cube_sync.patch)、[both_sync.patch](both_sync.patch)，
完整只读私有包路径见source。CPU依赖图解析、编译/load通过，定向Ruff、shell、diff通过。
全仓format.sh ci仍缺pre-commit，未声明通过。

原始报告在 `../results/hca_qgroups_sync_20260930/`；泳道为
`dfx/{base,both_sync}/dfx/merged_swimlane.json`。
`prepare.py`从0742f07c算子状态创建新命名副本，不得覆盖已冻结包；
CPU编译复用 `hca_residual_reuse_20260930/compile.py`，单卡排队执行
`task-submit --device auto --max-time 900 --run 'bash tests/pypto_test/hca_qgroups_sync_20260930/run.sh both_sync 131072 16'`。
只读汇总使用 `hca_mix_schedule_20260930/collect.py --experiment-root tests/pypto_test/hca_qgroups_sync_20260930`。
