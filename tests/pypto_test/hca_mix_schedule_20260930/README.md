# mHC mix与Sinkhorn同步联动（2026-09-30）

基线9c2ede34，CANN9.2 beta2、NZ2、atomic0/det0、BF16残差，正式第3层权重。
Native和CSA不变；两冻结候选只改变sync_start。mix为24个M4 worker、Sinkhorn12个M8，
8个KV预热worker按真实时序与前段重叠；并非把所有SPMD统一同步或扩核。

## 同进程ABBA

每侧10次真实设备span，单位μs；每候选使用自己的同卡baseline，不按不同轮绝对时间排序。

| 对照 | baseline min/max/mean | candidate min/max/mean | mean变化 |
| --- | --- | --- | --- |
| both_sync_h131072_b16 | 586.750/640.000/605.500 | 567.500/605.750/586.575 | -3.126% |
| mix_sync_h131072_b16 | 574.250/606.750/588.925 | 573.500/602.750/586.350 | -0.437% |

两候选四类完整输出/cache/state跨版本逐bit，自身eager/graph和保护区通过。
任务task_20260930_045354_367507923291退出0。原task_20260930_044651_357607519520
在解析history参数时退出2，没有进入计时；脚本已要求显式history/batch，避开队列附加参数。

## 联动证据与取舍

独立DFX task_20260930_045654_372396331433退出0。both_sync的mix组跨度29.76→30.92，
Sinkhorn35.98→36.22；Q反量化55.54→70.74，Attention起跑294.08→315.18μs。
该独立窗口没有显示整齐起跑直接缩短关键链，不能将ABBA收益归给单组等待减少。
ABBA的both_sync五个小组mean均下降，保留其整体调度线索；不把DFX和实际重放拼成同一测量。

随后与widen M4/K512做组合，直接对照M4/K512自身：长mean589.525→588.175，
短633.100→637.300；8:2相对mean约−0.050%，max分别−9.75/−19.50。
尚无稳定增量收益，暂不叠加生产；低max保留为削峰线索，未外推EP16收益。
组合完整结果见[下一轮](../hca_widen_box_20260930/README.md)。

## 复现与证据

- [冻结来源](source.json)、两个零上下文patch、compile_*.json。
- [完整汇总](summary.json)、[独立核时与组跨度](incore.json)。
- 原始报告：`../results/hca_mix_schedule_20260930/`。
- 泳道：`../results/hca_mix_schedule_20260930/dfx/{base,both_sync}/dfx/merged_swimlane.json`。
- `task-submit --device auto --max-time 900 --run 'bash tests/pypto_test/hca_mix_schedule_20260930/run.sh 131072 16'`
- CPU汇总：`python tests/pypto_test/hca_mix_schedule_20260930/collect.py`。

未新增本版本七档Native或整机token验收；20%目标未达成。
