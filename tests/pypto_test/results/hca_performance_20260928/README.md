# HCA 单卡七档 Native / PTO 初步性能对比

日期：2026-09-28。七个 task 均退出 0；状态为 MEASURED，Native 数值验收尚未通过。

本轮以中位数比较：仅 8K/B24 的 PTO 快 1.76%，其余六档慢 2.30%～10.12%。
这是首轮各 20 个样本的初步结果，1.76% 的小幅优势尚不作为稳定收益；不据此推断整模型收益。

## 测量口径

- 正式权重：`/data/model/DeepSeek-V4-Flash-0731-w8a8`；Native 第 3 层（C128 HCA）。
- 单卡 TP1，B 为请求数，每个请求 S6；输入 hidden 和历史 cache 为固定种子 20260928 的合成数据。
- 七档各用一张卡并行，每个 Native/PTO 对照在同一进程、同一张卡完成，物理卡见下表。
- A3，设备名 `Ascend910_9392`，CANN 9.0、torch/torch_npu 2.10；NZ mode 2，Native deterministic level 0，PTO atomic_add=0。
- 当前 Native 环境缺少 `TransposeBatchMatMulWeightNz`，wo_a 依照既有兼容逻辑使用 ND，其余权重遵循 NZ mode 2。
- 两侧范围相同：mHC pre + input norm + HCA attention + mHC post；包含本层 compact metadata 生成。FFN 不在范围内。
- PTO 测真实 `torch.ops.vllm.dsv4_hca_forward` 服务入口；先检查 eligible，禁止 fallback 冒充 PTO。
- 各侧预热 5 次、正式 20 次；每轮交替 A/B、B/A。图外 NPU Event 包住一次 ACL Graph replay，校验时间戳每次更新。
- 每次恢复同一份 cache/state 初态，恢复、输出毒化、初始化、编译、图捕获均不计时；无 profiler。
- 下表延迟变化 = (PTO P50 / Native P50 − 1) × 100%；正数表示 PTO 更慢。

## 设备耗时

| 历史 | Batch | Native P50 μs | PTO P50 μs | PTO 延迟变化 | Native P95 μs | PTO P95 μs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128K | 4 | 427.66 | 470.93 | +10.12% | 460.18 | 499.96 |
| 128K | 8 | 551.33 | 575.59 | +4.40% | 585.18 | 597.02 |
| 128K | 16 | 693.89 | 713.06 | +2.76% | 709.12 | 777.76 |
| 8K | 16 | 565.25 | 615.31 | +8.86% | 597.60 | 628.52 |
| 8K | 24 | 717.56 | 704.92 | -1.76% | 742.52 | 725.14 |
| 8K | 32 | 777.73 | 795.60 | +2.30% | 792.48 | 801.46 |
| 8K | 40 | 869.16 | 929.59 | +6.95% | 876.00 | 945.72 |

全部原始样本、均值、最小值和最大值保存在各档 report.json 中，不剔除慢样本。
事件测量覆盖完整设备区间，未用各 kernel 的时间加和代替延迟。

## 功能与数值

七档的 Native/PTO eager 保护区、只读 metadata、两侧计时图保护区、服务与直接调用一致性、
以及两侧图重放与各自 eager 的输出和完整 allocation 一致性均通过，每档 88 项。
各自 graph/eager 逐 bit 一致，不代表 PTO 与 Native 数值一致。

| 历史 | Batch | 输出 max_abs | 输出 RMSE | 输出不同元素 / 总数 | SWA 不同字节 | state 不同字节 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128K | 4 | 0.015625 | 0.00132080 | 74123 / 393216 | 3778 | 49167 |
| 128K | 8 | 0.031250 | 0.00132121 | 148097 / 786432 | 7376 | 96933 |
| 128K | 16 | 0.031250 | 0.00129920 | 290687 / 1572864 | 15092 | 194198 |
| 8K | 16 | 0.031250 | 0.00191865 | 477380 / 1572864 | 15027 | 194198 |
| 8K | 24 | 0.031250 | 0.00191347 | 716389 / 2359296 | 22513 | 291210 |
| 8K | 32 | 0.031250 | 0.00190404 | 949017 / 3145728 | 29969 | 388613 |
| 8K | 40 | 0.031250 | 0.00191885 | 1193579 / 3932160 | 37661 | 487989 |

Native/PTO 输出的非有限值为 0；零容差对照七档均有差异，尚不能宣布精度通过。
cache/state 使用完整原始 allocation 的字节比较，上表不同字节数不是 BF16/FP32 元素数。
七档 history 均为 128 的倍数，S6 本步没有跨 C128 组完成边界，compressed cache 不新增写入；
其两侧逐 bit 一致不能作为新压缩行数值通过的证据。跨界场景见先前 H124/H32764 的功能报告。
本轮不覆盖连续状态轨迹、padding/生命周期、P→D 离线 KV 或 DP/EP16 的整模型验证。

## 证据与复现

分支 `dsv4-flash-hca-pto-v0.25.1rc1`，基底 `65500b91` 加本地 HCA 接入；七档运行期间生产代码未改。
[命令清单](jobs.json)、[机器可读汇总](summary.json)、[CSV](summary.csv)。

| 历史 / Batch | 物理卡 | task | 原始报告 |
| --- | ---: | --- | --- |
| 128K / B4 | 0 | `task_20260928_140950_337067824293` | [report.json](h131072_b4/report.json) |
| 128K / B8 | 1 | `task_20260928_140952_33710043871` | [report.json](h131072_b8/report.json) |
| 128K / B16 | 2 | `task_20260928_140953_337161114136` | [report.json](h131072_b16/report.json) |
| 8K / B16 | 3 | `task_20260928_140954_337224419317` | [report.json](h8192_b16/report.json) |
| 8K / B24 | 4 | `task_20260928_140955_33729377829` | [report.json](h8192_b24/report.json) |
| 8K / B32 | 5 | `task_20260928_140956_33736332289` | [report.json](h8192_b32/report.json) |
| 8K / B40 | 6 | `task_20260928_140957_337433531868` | [report.json](h8192_b40/report.json) |

入口：`tests/pypto_test/run_hca_single_layer.sh`；计时实现：`tests/pypto_test/dsv4_hca_performance.py`。
命令必须通过 task-submit 提交；每档本地 states.pt 与编译产物保留，不提交二进制和大快照，不做 hash 校验。
