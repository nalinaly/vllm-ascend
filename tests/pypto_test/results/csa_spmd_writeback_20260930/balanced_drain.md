# SPMD同步协调的既有泳道证据

单位μs。跨调度线程取drain时间并集，不叠加prepare/publish子阶段。
此值是DFX中的协调区间，不等于设备空转或无profiler CSA的净开销；
相连区间数也不等于协议轮数。任务关联只由同轴时间观察，不能证明因果。

| 实验 | 档位 | 配置 | 两窗drain并集 | 平均 | 两窗相连区间数 |
| --- | --- | --- | --- | ---: | --- |
| csa_spmd_writeback_20260930/balanced_long | 131072/B16 | baseline_start | 19.16/21.10 | 20.13 | 2/2 |
| csa_spmd_writeback_20260930/balanced_long | 131072/B16 | wb16_sync | 20.22/32.46 | 26.34 | 2/3 |
| csa_spmd_writeback_20260930/balanced_long | 131072/B16 | wb16_pool_sync | 32.52/20.08 | 26.30 | 4/2 |
