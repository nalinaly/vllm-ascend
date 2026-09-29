# 长档attention的softmax统计留在UB

参考最新本地ops-transformer `28f40354` 中SWAVectorBlock的`softmaxMaxBuff`和
`softmaxSumBuff`：max/sum由同一AIV产生并在延后的PV归约阶段消费，使用UB环即可。
本候选只改变存储位置，保留每块max、BF16概率、raw/压缩块次序、running归约与FFTS握手。
scores/probs/values仍跨Cube/Vector走原来的GM交接，短档路径不改。

## 正确性修正

v1显式CPU编译通过，但设备跨版本检查失败，未进入性能计时。
生成码显示动态`tile.slice`在后续reshape中失去环槽偏移。v2改用显式Vec `tile.extract`，
生成动态`TEXTRACT`，完整输出/cache/state恢复逐bit一致。
详见[生成代码证据](COMPILER_SLICE_NOTE.md)，本轮没有修改编译器或运行时。

## 独立筛选与核内数据

CANN9.2、NZ2、atomic0、128K/B16，基底bef9f7fa，单位μs。
同进程每侧10次纯设备span：

| 实现 | min | max | mean |
| --- | ---: | ---: | ---: |
| base | 584.25 | 629.00 | 600.53 |
| v2 UB环 | 598.00 | 648.25 | 610.23 |

这轮整段慢9.70μs，没有整段收益。
独立DFX中attention AIV核内min/max/mean从186.78/192.60/189.74变为174.74/181.62/177.84，
mean下降6.27%、max下降10.98μs；AIC mean185.64→173.76，组跨度188.68→177.64。
生成AIV代码减少两处TLOAD和两处TSTORE，增加显式UB复制/提取；没有减少跨核同步次数。
未改的O-A/O-B核内也变快，gather组跨度却增加34.86→69.32，单张DFX不能唯一归因。
按用户要求保留明确的核内优化方向，同时保留整段变慢的证据并继续解决调度。

详细结果：[result_v2_h131072_b16.json](result_v2_h131072_b16.json)。
筛选任务`task_20260930_021255_114737517494`，DFX任务`task_20260930_021440_1180574330`。

## 与O-B的组合

冻结4ffe0a53作为base，叠加UB环作为候选。128K/B16与16384/B4均进入long分支，
后者的压缩页表形状为[4,6]，有效压缩历史只有128行，覆盖较短循环的排空路径。
完整输出、三组cache/state、eager/graph及保护区通过。
组合筛选任务`task_20260930_022008_122998616673`。

同进程B16均值577.20→596.85，观察到回退，故又用生产服务入口同卡ABBA核对；
两种测量独立报告，不把它们合并平均。正式服务入口结果由
[result_combo.json](result_combo.json)保存，任务`task_20260930_022146_12482866157`。
服务入口每侧18次设备span，base min/max/mean为585.75/638.50/600.86，
UB环为561.50/655.50/592.21：本窗口mean下降8.65μs（1.44%），max增加17.00μs。
两轮候选均值580.22/604.19，有明显漂移；与同进程筛选方向不同，尚不能认定稳定整段收益。
保留决策仅表示按既定要求留下核内收益，不代表Native或整模型验收通过。

## 复现文件

`prepare.py`保留v1复现；`prepare_v2.py`改为显式extract；`prepare_combo.py`冻结组合包。
三份CPU编译报告都单独保留。`run_swimlane.sh`与`run_combo.sh`必须经task-submit运行。
`collect.py`在CPU读取现有结果，检查完整状态，分别保存服务入口ABBA、同进程筛选与DFX。
不增加形状无关的重复测试，不测试未修改的short分支。
