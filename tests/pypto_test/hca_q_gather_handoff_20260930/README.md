# Q24真实满核后的消费者衔接（2026-09-30）

控制是第133节 `qr_late` 冻结候选，未接入生产。生产仍0742f07c。
该候选在独立泳道中让24份Q工作分布到24个Cube，最后Q提前50.46μs完成，
但反量化最终完成几乎不动；48份反量化分布到46个AIV、仍有两核重复。
compressed gather与反量化重叠。核内逻辑没有变化，问题转移到消费者的分配和衔接。

本轮两个候选都在同一控制上比较，不相加独立收益：

| 候选 | 修改 | 要验证的作用 |
| --- | --- | --- |
| cache_late | hca_norm_rope_write关闭allow_early_resolve | 减少其消费者gather在cache完成前预置，观察是否释放反量化的调度空间 |
| cache_late_dq_sync | 上项加四组各12-AIV反量化sync_start | 联动观察组内启动收拢和gather交错，是否缩短最后反量化完成时间 |

保留24个Q Cube、四组12个DQ AIV、48块上限的gather以及现有依赖边；
核内分块、stage2流水、算术、输出、cache/state不变，不把48份累计工作等价于48个同步物理核。
第112节旧单组48-AIV同步的回退不能直接替代当前四组结构的增量结论。

两个候选均已通过CPU依赖图、编译和load，私有源码已冻结。
128K/B16真实第3层ABBA、完整状态精确/重放/保护区检查已分别提交自动单卡队列，
截至[task_statuses.json](task_statuses.json)记录时仍pending，不能称为已测或接入。
只有真实收益证据出现后再扩受影响短档、七档和整机。

来源见[source.json](source.json)，`prepare.py`复制整包后只修改两个调度属性。
增量patch为零上下文，应用需`git apply --unidiff-zero`；不修改共享PyPTO/Simpler。
CPU复用`hca_residual_reuse_20260930/compile.py`，上卡复用`hca_pair_screen_20260930/run.sh`，
句柄见[tasks.json](tasks.json)，不得因观测超时重复提交仍存活任务。
