# Q分组生产者与并发任务联动（2026-09-30）

基线f14d8d90，生产算子与9015d45a相同；CANN9.2 beta2、NZ2、atomic0/det0、BF16残差、正式第3层。
沿用上一轮已消除reshape假依赖的二维私有Q；不改Native、共享CSA、工具链或权重/cache布局。
上游pypto-lib2164563仍是整份INT32投影后反量化，本候选按head组独立发布，区别和目的明确保留。

## 实测与取舍

同进程ABBA每侧10次真实设备span，单位μs；不同候选只与自己的base比较。

| 档位/候选 | base min/max/mean | candidate min/max/mean |
| --- | --- | --- |
| 128K/B16 两组，每组10 Cube/24 AIV | 572.500/609.250/587.225 | 587.250/614.250/597.375 |
| 128K/B16 四组，每组4 Cube/12 AIV | 585.250/611.000/593.800 | 563.750/579.000/571.475 |
| 8K/B24 四组，每组4 Cube/12 AIV | 608.750/662.500/626.075 | 610.750/641.750/627.825 |

两组版mean增加10.15，不接入。四组16 Cube长档mean少22.325（−3.760%），
该轮全部候选样本快于全部base样本；短档mean增加1.750（+0.280%）。
长短8:2加权约−2.952%，样本max分别少32.00/20.75，按用户口径保留四组16 Cube。
这仍不是七档或EP16验收，也不将单卡样本max等同多卡尾延迟已经解决。

## 泳道与因果边界

同卡独立DFX中，原Q Cube核内mean58.90，候选四组mean75.76–79.79；
减少worker后每worker工作更多，不能声称核内变快。四组Cube跨度77.86–93.10，
反量化实际重叠，最后在307.20完成；base反量化在304.18完成。
压缩gather跨度36.42→46.18，Attention AIV启动306.58→309.52，核内mean185.37→178.56。
所以本候选按无DFX的整段收益保留，不能将该窗口解释为Q关键链直接缩短22μs。
后续仍需联动生产者、并发gather及下游，不直接相加各task的变化。

## 接入与功能范围

- 生产新增HCA局部`q_projection_streamed.py`与`qkv_proj_rope.py`。
- 私有Q直接为[T,32768]且manual_dep=True，四组写各自head，四个TaskId直接作为长短Attention依赖。
- Q_B仍M128/N256/K256 stage2、INT32累加；反量化/RMS/RoPE算术和舍入顺序不变。
- ND通过带自动依赖的三维视图调用原公共Q投影；没有把NZ优化强加给ND。
- `ready`的函数内if/reshape未能推断旧Q helper元数据，CPU即失败；`ready_v2`在Python层选择
  明确签名的ND/NZ helper后，两种模式均完整编译/load通过，未改工具链。
- 长短四类完整状态跨版本逐bit、自身eager/graph及保护区通过。
- `ready_v2`的16507/B3包含2行尾，同址补位3→2→1→3通过；16384/B4 mode0的ND回退通过。
  这两项cycles=1仅作功能证据，不用于性能结论。
- 生产有效函数AST与已测ready_v2相同，仅移除未用局部变量/导入、整理格式和说明；
  保留原先未调用的legacy attention声明，避免扩大接口变更。

## 证据与复现

[source.json](source.json)、[summary.json](summary.json)、[incore.json](incore.json)、
[tasks.json](tasks.json)、[production_ast.json](production_ast.json)。
原始报告为`../results/hca_qb_groups_20260930/<side>_h<history>_b<batch>/report.json`；
泳道为`../results/hca_qb_groups_20260930/dfx/{base,flat4_c16}/dfx/merged_swimlane.json`。

`prepare.py`从上一轮冻结的streamed4_flat派生候选；`prepare_ready.py`生成ND回退明确的可接入包。
CPU显式编译复用`hca_residual_reuse_20260930/compile.py --operator-source ... --nz-mode 0|2`。
排队入口：`task-submit --device auto --max-time 900 --run 'bash tests/pypto_test/hca_qb_groups_20260930/run.sh flat4_c16 131072 16'`。
只读汇总：`python tests/pypto_test/hca_mix_schedule_20260930/collect.py --experiment-root tests/pypto_test/hca_qb_groups_20260930`。
输出目录禁止覆盖；重测必须另取名字。未新增本候选七档Native或整模型token结论。
定向Ruff、shell语法和diff检查通过；全仓format.sh ci仍因缺pre-commit未通过。
