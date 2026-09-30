# 四组Q提前解析与消费者联动（2026-09-30）

基线是0742f07c加冻结的online-softmax候选，**不是新生产版本**；两候选只改Q任务提前解析。
保持四组各4 Cube/12 AIV、核内tiling、sync_start、算术和TaskId。旧20 Cube整组实验不能
替代当前四组发布结构的对照。CANN9.2/NZ2/atomic0/det0、正式第3层、BF16残差。

考虑的联动：Q生产者提前解析能让反量化更早获取任务，但也可能使消费者提前占用AIV、
影响压缩gather和Attention；同时测试关闭反量化提前解析是否能减轻这一影响。
本轮只得到完整区间证据，没有新DFX来认定某个竞争链是回退的唯一原因。

128K/B16同进程ABBA各10次实际span，μs，min/max/mean：

| 候选 | 自己的online控制 | 候选 |
| --- | --- | --- |
| Q Cube提前解析 | 547.750/596.000/563.125 | 559.500/587.250/572.075 |
| Q提前＋反量化延后解析 | 583.250/630.500/597.975 | 588.250/627.500/607.650 |

mean分别增加8.950（+1.589%）和9.675（+1.618%）；两轮均只有首个ABBA小组获益，
其余四组回退。P50分别增加14.625和10.625。虽然max少8.75/3.00，但控制max均位于
整个profile的首个重放；不将它声明为稳定削峰。原始慢点没有删除。

输出、三类完整cache/state跨版逐bit，各自eager/graph及保护区通过。
两个候选均不接入、不扩短档/七档/模型测试，不把不同窗口的绝对值用于候选间排序。
它们与online基线的组合增量已实际测量，没有把独立优化百分比相加。

[source.json](source.json)、[summary.json](summary.json)、[tasks.json](tasks.json)、
[cube_early.patch](cube_early.patch)、[cube_early_dq_late.patch](cube_early_dq_late.patch)。
CPU解析/编译/load通过；私有包只读。全部NPU通过task-submit自动排队，使用公共pair入口，
源码路径见source；只读汇总使用`hca_mix_schedule_20260930/collect.py --experiment-root`。
生产仍0742f07c，最新Native对照仍日志123节，本轮未证明领先Native20%。
定向Ruff、shell语法、diff检查通过；全仓`format.sh ci`缺pre-commit，未宣称通过。
