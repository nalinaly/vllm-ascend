# HCA 长档双 query 复用实验（2026-09-30）

三个设备候选均未采用：完整 HCA 与 Attention 核内均未受益。生产仍为 ad0e6bbe。
保留实现、容量失败与完整证据，不能据此宣称所有多 query 复用方式均不可行。

本地 ops-transformer 28f40354 的 `sparse_attn_sharedkv_tiling.cpp:1538` 在 CFA/SWA
路径按 `(256/gSize)*gSize` 组织 M，提供多 query 共用 KV 的参考；未证明安装包本档采用此模板。
其 arch22 `sparse_attn_sharedkv_swa_block_vector.h:757` 在多个块之间通过 GM 保存 FP32 累积结果。
PTO 原实现每 query M64、三槽 KV、跨 query 延续两步预发射、FP32 输出留 UB。

候选将 S6 中相邻两条 query 的压缩部分合为 M128，共享 KV；raw 部分仍各自 M64，
保留各自滑窗。每 query 的压缩块次序、块内最大值、BF16 概率舍入、sink 和 RoPE 均不变。
仍为 24 个 MIX 核组，不改变上游 gather、任务依赖或短档分支。

| 候选 | 累积结果位置 | KV 流水 | 编译静态 Vec 上限 | Mat 上限 |
| --- | --- | --- | ---: | ---: |
| pair_v2 | 全部 GM | 两槽、提前一步、每 pair 排空 | 117504 | 425984 |
| pair_hybrid_reload | 前半 UB、后半 GM；最终按 H16 读回 | 同上 | 174848 | 425984 |
| pair_three_slots | 全部 GM | 三槽、提前两步；Q 每块重载，与 P 复用 L1 | 118016 | 524288 |

单位字节。全 UB 版本编译需求 240384，超过当前 PyPTO 配置的 188416；未提交 NPU。
初版半驻留也因最后两片活跃期延长达到同一峰值，改成最终 H16 读回后编译通过。
这描述当前分配器配置，并非声称 A3 物理 UB 只有 184 KiB。

## 性能与联动

CANN9.2、NZ2、atomic0、det0，正式第3层权重；128K/B16，同进程 ABBA 每侧10次真实设备 span。
npugraph_ex dynamic=False、inplace/static 开启、PTO SuperKernel 关闭。单位 μs：

| 候选 | 本轮 base min/max/mean | 候选 min/max/mean |
| --- | --- | --- |
| pair_v2 | 583.00/627.00/596.40 | 599.50/631.50/613.45 |
| pair_hybrid_reload | 558.50/602.00/573.25 | 563.00/596.25/578.95 |
| pair_three_slots | 577.50/611.50/590.08 | 604.25/631.75/618.28 |

半驻留样本 max 下降5.75，但 mean 增加5.70，不作为稳定削峰证据。
不同轮次各有控制，不以绝对值横比三个候选。

独立 DFX 的 Attention AIV 核内 min/max/mean：

| DFX 轮次 | base | 候选 |
| --- | --- | --- |
| pair_v2 | 163.20/166.98/164.73 | 174.06/187.16/182.73 |
| 半驻留 | 159.36/164.88/161.66 | 186.28/194.96/191.23 |
| 三槽 | 同一轮 base | 197.22/210.08/202.89 |

Attention AIC mean 也分别160.55→174.26、157.30→183.30/191.62。
半驻留与三槽的 Attention 更早启动（347.32→318.00/314.44），但结束为512.48→513.22/524.72。
Q、反量化和 gather 的启动/跨度也改变：更早开始不代表核内变快，不能把整段效果只归于 KV 复用。
全部任务起止及核内 min/max/mean 见 [incore_gm.json](incore_gm.json)、
[incore_refinements.json](incore_refinements.json)。未使用跨轮次相减作因果结论。

四类完整输出/cache/state 跨版本逐 bit、自身 eager/graph 与保护区通过。
只覆盖128K/B16；没有宣称尾块、短档、七档、Native 精度或整模型 token 通过。
未扩大无收益候选测试。

## 证据与复现

- [summary.json](summary.json)：三个同进程报告的来源、配置、状态检查与汇总。
- [codegen_memory.json](codegen_memory.json)：编译分配边界，不是运行时峰值监控。
- `prepare.py --variant pair_v2`：复制冻结基线并替换长档 attention；`attention_body.py.inc` 为实现片段。
- `prepare_ub.py`、`prepare_hybrid.py`、`prepare_three_slots.py`：后续局部方案及 patch。
- `compile_*.json`：两入口依赖图解析、测试入口显式编译/load；不是仅 import 的检查。
- `run_refinements.sh`、`run_dfx*.sh`：设备命令必须通过 task-submit 自动分配卡。
- `collect.py`：从现有性能报告与泳道重建汇总，无需占卡。

终态均 exit=0：v2 筛选 `task_20260930_034959_272162424011`、v2 DFX
`task_20260930_035216_276184729721`、两精化版筛选 `task_20260930_035933_286386020359`、
精化版 DFX `task_20260930_040233_288794123626`。

原始根目录：`tests/pypto_test/results/hca_pair_query_20260930/`。
性能：`{pair_v2,pair_hybrid_reload,pair_three_slots}_h131072_b16/report.json`。
泳道：`h131072_b16/swimlane_{base,pair_v2}/dfx/merged_swimlane.json`，
`dfx_refinements/swimlane_{base,pair_hybrid_reload,pair_three_slots}/dfx/merged_swimlane.json`。
