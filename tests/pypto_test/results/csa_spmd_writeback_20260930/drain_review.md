# SPMD同步协调的既有泳道证据

单位μs。跨调度线程取drain时间并集，不叠加prepare/publish子阶段。
此值是DFX中的协调区间，不等于设备空转或无profiler CSA的净开销；
相连区间数也不等于协议轮数。任务关联只由同轴时间观察，不能证明因果。

| 实验 | 档位 | 配置 | 两窗drain并集 | 平均 | 两窗相连区间数 |
| --- | --- | --- | --- | ---: | --- |
| csa_spmd_writeback_20260930/summary | 131072/B16 | baseline_start | 18.70/19.60 | 19.15 | 2/2 |
| csa_spmd_writeback_20260930/summary | 131072/B16 | wb48 | 17.10/17.54 | 17.32 | 2/2 |
| csa_spmd_writeback_20260930/summary | 131072/B16 | wb48_sync | 43.26/32.58 | 37.92 | 4/3 |
| csa_spmd_writeback_20260930/summary | 8192/B24 | baseline_start | 0.00/0.00 | 0.00 | 0/0 |
| csa_spmd_writeback_20260930/summary | 8192/B24 | wb48 | 10.48/0.00 | 5.24 | 1/0 |
| csa_spmd_writeback_20260930/summary | 8192/B24 | wb48_sync | 8.44/10.14 | 9.29 | 1/1 |
| csa_spmd_kv_20260929/m16_long | 131072/B16 | baseline_start | 19.14/18.16 | 18.65 | 2/2 |
| csa_spmd_kv_20260929/m16_long | 131072/B16 | kv_m16 | 18.32/18.70 | 18.51 | 2/2 |
| csa_spmd_kv_20260929/m16_long | 131072/B16 | kv_m16_sync | 29.00/30.70 | 29.85 | 3/3 |
