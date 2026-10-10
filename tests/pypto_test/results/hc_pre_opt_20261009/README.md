# 独立 HC_pre 优化证据（2026-10-09～10）

本目录对应 [优化过程日志](../../HC_PRE_OPTIMIZATION_LOG.md)，保留本轮候选源码和可读报告。
当前正式实现为 `vllm_ascend/ops/pypto/hc_pre.py`，与 `kernels/cv_shape_dispatch_v4.py` 一致。
历史候选仅用于追溯；正式入口不会导入本目录中的源码。

## 内容与状态

- `kernels/`：118 份完整独立候选源码，包含未采用的实验。
- 各实验子目录的 `report.json`：205 份报告，77 份 PASS、83 份 COMPILED、45 份 FAIL。
- `experiments.json`：实验汇总，保留原始报告路径和已完成场景。
- `sinkhorn_fixedpoint_cpu.json`：固定点分析的 CPU 证据，不能替代真机验证。
- `artifact_manifest.json`：逐文件路径及字节数，以及只保留在本地的目录说明；没有 hash 校验。
- 相邻 [hc_pre_profiles_20261010](../hc_pre_profiles_20261010/README.md)：14 场景、42 份三步 trace，
  包含 AscendC PyTorch profiling、PTO L1 和 PTO L4；另附四份来源报告及 v3/v4 源码快照。

COMPILED 仅表示生成并编译设备代码，FAIL 可能包含失败前已完成的场景，不得据此称整轮通过。
早期少数实验使用 release 基线或未同时开启外部 profiler，限制见过程日志和每份报告。
报告内的绝对路径是当时机器上的运行位置，源码、报告和可读 trace 已在本仓库归档。

## 最终验证链

| 报告目录 | 任务 | 范围 | 结果 |
| --- | --- | --- | --- |
| `cv_shape_dispatch_v3_all14` | `task_20261010_092534_131182121051` | 七档 BS × S5/6，14 场景，L1 | PASS |
| `cv_shape_dispatch_v4_large` | `task_20261010_093010_15801491533` | B24/B28 × S5/6，4 个大形状增量场景，L1 | PASS |
| `cv_shape_dispatch_v3_L4_export` | `task_20261010_095015_28703115564` | B2/4/8/16/20 × S5/6，10 场景，L4 | PASS |
| `cv_shape_dispatch_v4_L4_export` | `task_20261010_095016_28715337150` | B24/B28 × S5/6，4 场景，L4 | PASS |

v4 保留三个分块配置，在四个动态 T 范围中选择。相对 v3 未改分支的生成计算体已经直接文本比对一致，
只对变化的大形状范围增量测试。没有将 v3 全量与 v4 增量写成 v4 的同轮全量性能验收。
当前十四场景最新 L1 中位数为 27.64～39.96 μs；仅前四场景三次均低于 30 μs，完整目标尚未达到。

数值门禁保持 mixed atol/rtol=0.01/0.01、post/comb=1e-5/1e-4；
检查三输出、graph/eager 严格一致、同址更新输入、NaN 毒化输出和保护行，并核对实际执行分支。
使用正式第 2 层 HC 权重和合成 BF16 输入，范围为独立 HC_pre，尚不代表整模型 token 或性能验收。

## 复现及边界

构建指定 AscendC 基线及单卡运行命令见 [独立入口说明](../../HC_PRE_STANDALONE.md)。
测试通过 `--kernel-file` 选择某个完整候选，通过 `--swimlane-level 1` 或 `4` 分别采集。
PTO 计时取全部 AIC/AIV 的最早 incore 开始至最晚结束，包含内部转换、同步和等待；
排除首尾之外的 AICPU 区间，不用并行区间并集或核时长累加代替。
Native 取 HcPre 设备事件。L4 为独立采样，额外记录可能改变耗时，性能表使用 Native 与 L1。

大体积原始采集、编译产物、设备二进制、模型权重及下载 ZIP 留在本地，路径见清单。
本次不重跑真机性能测试，继续使用已完成验证；六项 CPU 计时边界测试在 L4 支持修改后已通过。
