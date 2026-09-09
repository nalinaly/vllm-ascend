# CSA profiling 文件索引

更新时间：2026-09-08 10:14:08

测试口径：A3 device0、TP1、B4/S8、C8191、TRB L1 ACLGraph；真实生产
`dsa_forward` 入口，两路合成非零输入/权重一致，六类可变 cache 地址独立。
历史 cache 初始多数为零，不能外推真实 serving 吞吐或整网问答精度。

## 先看这两个文件

- [native_device_trace.json](native_device_trace.json)：6 次 replay，38 kernel/次，
  含实际 device stream、事件同步、DMA 和 MODEL 控制记录。
- [pypto_device_trace.json](pypto_device_trace.json)：同一诊断进程、同一配置的
  6 次 L1 replay，含 AICPU 与融合 AICore **父 kernel**。
  **当前还没有逐 core 子任务泳道。不能以此分析 child-task 空洞或声称交付齐全。**

可将 JSON 拖入 Perfetto 或 Chrome tracing。保留原始绝对 device 时间戳，
两文件中的不同 replay 之间有真实空隙，没有重新拼接时间线。

## 预览与明细

| 文件 | 内容 |
| --- | --- |
| [native 预览 PNG](native_device_preview.png) / [SVG](native_device_preview.svg) | 固定第 3/6 个 replay，数字对应 CSV 的 Replay Kernel |
| [PTO 父 kernel 预览 PNG](pypto_device_preview.png) / [SVG](pypto_device_preview.svg) | 固定第 3/6 个 replay，不是 child-task 泳道 |
| [native CSV](native_kernel_details.csv) | 228 行，原始 profiler 列 + 设备归属和 replay 编号 |
| [PTO CSV](pypto_kernel_details.csv) | 12 行，保留原始 kernel 名称和实际时长 |
| [诊断 replay 汇总](diagnostic_replay_summary.csv) | MODEL span、kernel span/sum/union/overlap/idle |
| [正式延迟汇总](formal_latency_summary.csv) | 3 个独立进程，每路每进程 100 个关闭 profiler 的样本 |
| [全部正式 device 样本](formal_device_samples.csv) | 600 行，无长尾剔除；含 ABBA 位置 |
| [审计清单](audit_manifest.json) | 覆盖率、原始输入与审计代码 SHA256 |
| [软件基线](software_manifest.json) | 三仓 commit、CSA 源码及 TRB 二进制 SHA256 |

CSV 自定义 kernel 的原始 `Model ID=4294967295` 未改写；新增列
`Resolved Device Model ID` 来自 device trace 的精确 timestamp+name 对应。
不能按原始 CSV Model ID 直接删掉这些行。

native 的 kernel_sum 是多 stream 时长相加，不等于延迟；PTO 的
kernel_overlap 是父 AICPU 与 AICore 重叠，不代表 child-task 并行收益。
带 profiler 的诊断时长不进入正式 p50/p90/p99。

[完整定位记录](../../../20260908_apple_to_apple_profiling.md)；
[可重跑离线审计本](../reproduction_audit.ipynb)；
[fixture 身份核验](../identity/fixture_identity.json)。

离线审计本已从头到尾执行并通过结构校验。PTO 逐 core 子任务泳道仍是未完成项，
需要在独立诊断分支给当前 L1 runtime 接入采集支持。
