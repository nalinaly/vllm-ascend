# Q提交粒度与实际占核：先核实分配，再做同步联动（2026-09-30）

以5c0fa709记录的Q24候选为控制；生产仍是0742f07c，二者不可混称。
CANN9.2/NZ2/atomic0/det0、正式第3层、BF16残差，128K/B16；npugraph_ex dynamic=False，
inplace/static开启、PTO SK关闭，计完整mHC pre/norm/HCA/O/post。

## 已完成的提交粒度对照

Q Cube总数保持24、M128/N256/K256、stage2，四组12-AIV反量化保持。
原版四份INT32投影改成一份64 heads或两份32 heads，消费者通过完整Tensor及显式head偏移读取，
没有使用丢行stride的子视图。Q输出仍是二维manual_dep Tensor，四份TaskId直接连到Attention。
延后部分Q发布是实际代价，不能把减少提交次数直接算成收益。

同进程5次预热、ABBA五组，每侧10个实际span。μs，min/max/mean；每行只与自己的控制比较。

| Cube提交 | 原四组Q24 | 候选 |
| --- | --- | --- |
| 一组24 | 557.500/591.750/575.025 | 563.000/591.500/571.800 |
| 两组各12 | 546.000/574.750/554.925 | 555.000/576.000/563.275 |

两组版mean多8.350，五个ABBA小组全部回退；不接入。
一组版mean少3.225、P50少4.50，但五个小组只有三组改善，首组差异最大；暂不接入。
两版完整x_out、swa、compressed、state跨版逐bit，自身图重放与保护区通过。

## 独立同卡DFX及调度机制

四组控制中24份工作用了21个物理Cube；一组24中只用了19个Cube，5个核各执行两份，
另5个核没有执行Q。这不是“写了24 blocks就已经用满24核”。
[q_placement.json](q_placement.json)保存每份工作对应物理核及实际起止。

| 项目，μs | 四组Q24 | 一组24 |
| --- | --- | --- |
| Q组0核内mean/跨度 | 50.867/105.70 | 50.675/115.60（整组） |
| 最后一份Q Cube完成 | 267.78 | 267.52 |
| 最后反量化完成 | 312.54 | 315.22 |
| Attention AIV开始/结束 | 317.46/506.90 | 319.20/507.18 |
| compressed gather跨度 | 49.72 | 35.22 |

仅合并任务并未消除二次占核，也未提前关键链；DFX窗口与上表计时分开解释。
本地Simpler的`scheduler_dispatch.cpp`约620–715、735–809行说明，线程根据自己的idle/pending核
领取SPMD范围，再把未领完部分放回共享队列；任务的逻辑block数不是物理核一一映射保证。
这里只据此提出预置不均的解释，尚不声称已唯一定位运行时缺陷，也没有改共享Simpler。

## 两个针对性后续对照

- `whole_sync`以一组24为父版本，仅开启其sync_start；生成码为24 blocks和require_sync_start=true。
- `qr_late`以原四组Q24为父版本，仅关闭上游`qr_rms_norm_quant`的allow_early_resolve。
  该开关决定其消费者能否提前预置；不同于之前修改Q Cube自身开关来影响反量化。

两版CPU依赖图、编译及load已通过，真机状态以[tasks.json](tasks.json)里的句柄查询为准。
不要将CPU编译通过当成功能/性能通过；未根据假设替换生产。后续结果追加于下方。

## 证据与复现

[source.json](source.json)、[summary.json](summary.json)、[incore.json](incore.json)、[tasks.json](tasks.json)。
patch为零上下文，应用到对应父版本需`git apply --unidiff-zero`；私有包只读，不覆盖运行源码。
CPU复用`hca_residual_reuse_20260930/compile.py`，NPU复用`hca_pair_screen_20260930/run.sh`，
DFX复用`hca_mix_schedule_20260930/run_dfx.sh hca_qqueue_groups_20260930 base cube_groups1`。
全部NPU经task-submit自动排队。泳道位于
`../results/hca_qqueue_groups_20260930/dfx/{base,cube_groups1}/dfx/merged_swimlane.json`。
不扩无明确收益组合的七档或16卡测试。
# 后续完成：QR预置与反量化衔接

whole_sync和qr_late两任务均exit0，完整输出/cache/state跨版逐bit，图重放和保护区通过。
CANN9.2/NZ2/atomic0/det0、128K/B16，ABBA每侧10次，μs，min/max/mean：

| 增量 | 自己的控制 | 候选 |
| --- | --- | --- |
| 一组24 Cube整体sync | 568.500/612.250/583.675 | 571.500/592.000/581.825 |
| 四组Q24，仅QR不提前解析 | 584.000/625.000/602.700 | 588.250/605.750/593.675 |

whole_sync五组只有两组改善、P50反而580.125→582，mean/max受首组控制慢点影响，不接入。
qr_late四组改善，但前两组贡献最大，不能直接按−9.025μs接入生产。

新增同卡DFX（base_qr_followup→qr_late，另一个窗口，不与旧base拼接）显示：
Q24从20个物理Cube、4核重复工作，变为24个各一次；最后Q完成275.34→224.88μs。
但是48份反量化从39个物理AIV变为46个，仍有2核重复，最终反量化完成318.44→318.92；
Attention AIV启动322.82→320.34，只提前2.48μs。compressed gather与其重叠，
核内mean19.212→24.423、跨度54.98→39.94，不能简单把任何单项差值当作完整HCA收益。

保留Q24满核的调度候选，下一项联动gather生产者和四组反量化，不只继续修Q启动。
逐核记录见[qr_placement.json](qr_placement.json)，完整泳道在
`../results/hca_qqueue_groups_20260930/dfx/{base_qr_followup,qr_late}/dfx/merged_swimlane.json`。
