# Q生产者与反量化消费者的提前解析联动

基线a7a9f314，CANN9.2、NZ2、atomic0、det0，128K/B16，正式第3层权重。
两个候选保持核数、tiling、算术和依赖，只调`allow_early_resolve`：
`late_dq`关闭Q反量化提前解析；`early_q_late_dq`再同时开启Q投影提前解析。

同进程同卡ABBA，每侧10次真实设备span，μs，min / max / mean：

| 候选 | 自己的base | 候选 |
| --- | --- | --- |
| late_dq | 547.25 / 592.50 / 562.60 | 561.00 / 600.25 / 582.85 |
| early_q_late_dq | 587.50 / 634.00 / 602.73 | 583.25 / 622.75 / 603.90 |

前者mean/max都退化，不接入。联动版mean基本持平、样本max下降11.25μs；
按用户要求保留削峰线索，但还不能称稳定收益。不跨两轮的绝对值比较两个候选。
各候选四类完整输出/cache/state跨版本逐bit一致，自身图重放和保护区通过。
任务`task_20260930_030049_186179632360`退出0；完整报告和路径见[result_summary.json](result_summary.json)。

私有包由`prepare.py`生成，CPU显式解析/编译/加载通过后才执行`run.sh`。
初次prepare在Ruff失败后没有生成包，两次导入因此失败；修复后重新prepare和编译成功。
这些失败仅在CPU，不与正式设备计时混用。没有新Native、短档或整机验收结论。
