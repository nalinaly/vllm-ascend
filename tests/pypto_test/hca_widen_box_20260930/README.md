# HCA widen有效M4与调度组合（2026-09-30）

## 实现与依据

基线9c2ede34，CANN9.2 beta2、NZ2、atomic0/det0、正式第3层权重、BF16残差。
参考ops-transformer28f40354的
`experimental/mhc/mhc_pre_sinkhorn_premix/op_kernel/mhc_pre_sinkhorn_premix_m_split_core.h:120–198`：
BF16加载/转FP32/写中间值，并按分段次序累加统计；不能据此断言安装Native使用该模板或相同tiling。

旧widen按8行分工；新版本用8行物理盒保持FP32列向量32B对齐、每worker只处理4个有效token，
B16为12→24个worker，最多48个并循环覆盖更多token。显式Tile有效形状覆盖所有GM读写，
去掉共享单尾块暂存及完整/尾行分支，防止相邻worker的物理盒覆盖其他有效行。
K512版保持stage4及逐512列累加、高精度rsqrt；K1024版stage2但仍分两段512顺序统计。
没有改变下游数据布局、BF16舍入、依赖、Native或工具链。

## 同进程ABBA

每侧10次实际设备span，μs。linked_delta的baseline是M4/K512，其余是9c2ede34；
因此不能把两个差值直接加减，也不能跨卡比较不同轮的绝对值。

| 对照 | baseline min/max/mean | candidate min/max/mean | mean变化 |
| --- | --- | --- | --- |
| linked_delta_h131072_b16 | 575.500/621.000/589.525 | 575.500/611.250/588.175 | -0.229% |
| linked_delta_h8192_b24 | 612.750/670.500/633.100 | 621.750/651.000/637.300 | +0.663% |
| m4_k1024_h131072_b16 | 571.250/623.750/593.950 | 575.000/618.000/590.450 | -0.589% |
| m4_k512_h131072_b16 | 576.750/624.500/590.250 | 555.250/603.750/568.650 | -3.659% |
| m4_k512_h8192_b24 | 610.000/674.000/621.875 | 620.750/643.750/631.250 | +1.508% |

M4/K512长档五个ABBA小组mean均下降，长mean−3.659%、短+1.508%，8:2加权−2.626%。
max分别−20.75/−30.25μs；只是本窗样本最大值，不声称已解决EP16长尾。
短档均值代价明确记录，按用户8:2总体有效即保留的口径接入；没有声称所有档位均改善。
K1024仅−0.589%，核内也更慢，未采用。linked同步组合均值增量约−0.050%，暂不接入；
它的max下降保留为后续联动线索，不能叠加两轮独立收益。

## 泳道：不是widen核内变快

DFX task_20260930_045538_370356413396同卡依次base/M4K512/M4K1024，退出0。

| 128K/B16任务或任务组 | base | M4/K512 | M4/K1024 |
| --- | --- | --- | --- |
| widen核内mean | 25.550 | 26.299 | 29.130 |
| widen组跨度 | 27.100 | 30.580 | 32.120 |
| linear核内mean | 14.058 | 16.713 | 14.339 |
| mix核内mean | 27.473 | 27.297 | 28.260 |

M4/K512前段并未变快；同一独立窗口里，Q反量化跨度75.20→61.06、压缩gather
72.64→46.74、Attention AIV跨度187.66→182.40。上游与下游的联动同时改变，
这些是观察，不是唯一原因证明；不能把各task差值相加当作完整HCA收益。
此候选按真实整段收益保留，而不是作为“扩核必定减少核内时间”的证据。

## 验证范围与任务

四类完整输出/SWA/compressed/state跨版本逐bit，自身eager/graph及保护区通过。
16507/B3包含4行worker的2行尾；同址补位3→2→1→3通过，cycles=1只作功能证据。
没有新七档Native或整机token验收。生产仅复制已测M4/K512的decode_hca.py，另做显式CPU编译/load。

| 任务 | 范围 | 结果 |
| --- | --- | --- |
| task_20260930_045354_367526922634 | M4/K512 128K/B16 | exit0 |
| task_20260930_045354_36751398870 | M4/K1024 128K/B16 | exit0 |
| task_20260930_045538_370372523596 | M4/K512 8K/B24 | exit0 |
| task_20260930_045654_372402618937 | M4/K512 16507/B3和补位图 | exit0 |
| task_20260930_045726_37342998885 | M4/K512→同步组合 128K/B16 | exit0 |
| task_20260930_045726_37342249954 | M4/K512→同步组合 8K/B24 | exit0 |

## 复现与证据

- [冻结来源](source.json)、三个patch、compile_*.json；linked patch在M4/K512之上应用。
- [完整汇总](summary.json)、[独立泳道核时](incore.json)。
- 原始报告：`../results/hca_widen_box_20260930/`。
- 泳道：`../results/hca_widen_box_20260930/dfx/{base,m4_k512,m4_k1024}/dfx/merged_swimlane.json`。
- `task-submit --device auto --max-time 900 --run 'bash tests/pypto_test/hca_widen_box_20260930/run.sh m4_k512 131072 16'`
- CPU汇总：`python tests/pypto_test/hca_mix_schedule_20260930/collect.py --experiment-root tests/pypto_test/hca_widen_box_20260930`。

输出目录禁止覆盖；重测应使用新目录。生产接入不包含同步组合或K1024。
本轮没有降低验收目标，HCA各档领先Native20%以上仍未达成。
