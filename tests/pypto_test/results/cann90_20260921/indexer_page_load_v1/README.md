# Indexer 页加载优化：2026-09-22

B40/S6/history=131073，同一张逻辑卡 8，同一份参考层权重、输入与初始状态。
优化生产源码仅涉及 `decode_indexer.py::indexer_score_topk_forest`。

| 指标 | 优化前 | 优化后 |
| --- | ---: | ---: |
| Indexer score leaf AIC 平均执行时间，独立 DFX | 28173.50 μs | 5701.32 μs |
| Indexer score leaf AIV 平均执行时间，独立 DFX | 28170.78 μs | 5706.04 μs |
| Top-K query merge 平均执行时间，独立 DFX | 35.96 μs | 35.20 μs |
| 完整 CSA 加 Native metadata，3 次 graph profile 中位数 | 31145.62 μs | 13522.88 μs |
| 同条件 Native 中位数 | 2473.26 μs | 2479.02 μs |

Indexer AIC 耗时下降约 79.76%，约 4.94 倍加速；完整 CSA 中位数下降约 56.58%，
约 2.30 倍加速。当前 PTO 中位数仍为 Native 的 5.45 倍。
PTO 三次设备跨度分别为：旧版 `[56087.22, 31145.62, 31018.80]` μs，
新版 `[14332.70, 13522.88, 8307.74]` μs；保留全部样本，使用三次中位数。
DFX 是另一个进程的一次 eager 诊断窗口，不与 graph profile 的时间直接混算。

## 图与原始证据

- [优化后 PTO 泳道图](profile/swimlane/merged_swimlane.json)
- [优化前 PTO 泳道图](../csa_profile_v1/swimlane/merged_swimlane.json)
- [优化后 PTO PyTorch profile](profile/pypto/pypto_profiling.json)
- [同次 Native PyTorch profile](profile/native/native_profiling.json)
- [优化前 PTO PyTorch profile](../csa_profile_v1/pypto/pypto_profiling.json)
- [完整前后对比](summary.json)、[本次 Native/PTO 比较](profile/comparison.json)
- [数值检查与优化前后逐 bit 对照](numerical_comparison.json)
- [生产源码快照](sources/vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/decode_indexer.py)
- [实际生成的 PTO 与 C++](generated/)
- [可下载的三份新 trace 与结果包](DSV4_CSA_INDEXER_PAGE_LOAD_profiles.tar.gz)

三个 trace 可以在 Chrome tracing 或 Perfetto 导入。泳道名称来自本次 DFX 实际编译的
`_jit__decode_csa_tp1_attention_i668rr27/kernel_config.py`，不是根据任务顺序猜测。
2549 个设备切片、1265 个 AICore block 和原始依赖保留；离线补名称不改变任何时间戳。
查看 `Worker View` 的 `indexer_score_topk_leaf_aic/aiv`；`Scheduler View` 还包含依赖等待。
原来的 query merge 长条主要等待 score leaf，merge 自身仅约 36 μs。

## 原因和修改

Native 每页有 4096 字节 INT8 K 和 64 字节 FP16 scale，本次实际 page stride 为 4160 字节。
旧实现为适配这个共享页布局，每页 K 使用 32 次 `gather_row([1,128])`。
每个 384-token score tile 共 12 页，需要 384 次小 GM→L1 读取；参考库的独立连续 K
布局则能按 `[32,128]` 整页加载。

新版在同一个 score leaf 内用两个 AIV 各读取 6 页，每页一次 `gather_row([1,4096])`，
写入 24 KiB 的片上 UB。UB 零拷贝 reshape 成 `[192,128]` 后通过 `aic_gather` 的
UP_DOWN 模式交给 AIC，恢复 `[384,128]` 的原始 INT8 MatMul 输入。
每个 score tile 的 K 读取由 384 次降至 12 次，GM 读取字节数仍为 49152 字节；
新增一次 UB→L1 片内传递。K 和 scale 继续直接使用同一个 Native allocation，
沿用真实 page stride、页表、非零 storage offset 和原来的尾页选择。
没有增加 GM 搬运缓冲、外部适配调用或 CSA 入口参数。
INT32 点积、FP16 转换、权重归约和 Top-K tie 顺序保持原实现。

首次尝试将页内 slice reshape 为 GM `[32,128]` 后直接整页 gather，CPU lowering 失败：
当前 PyPTO 会先将这个 slice 转成 Tile，`tile.gather_row` 的源必须为 GM Tensor。
该尝试的日志保存在 `failed_gm_slice/`，没有上卡。最终方案使用现有合法片内接口；
没有修改 PyPTO、Simpler、PTO-ISA 或 pypto-lib。

## 验证与复现

CPU 完整 CSA lowering、PTOAS 编译通过。两个真机完整对照共 36 项检查全部通过：

| 用例 | 任务 | 结果 |
| --- | --- | --- |
| B4/history131071 | `task_20260922_200911_325781123742` | 18 项 PASS |
| B40/history131073 | `task_20260922_200912_325787534` | 18 项 PASS |
| B40 Native/PTO profile 与独立 DFX | `task_20260922_201208_404184422233` | 三个进程 PASS |

前两项的输出、Top-K、scores 与 `adapter_redundancy_v4` 逐 bit 相同；六项 cache/state
及两侧保护区检查继续通过。本次 profile 的 Native、PTO、DFX 输出也分别与旧 profile
逐 bit 相同；PTO graph 与 eager 一致。每次重放仍只有一次 PTO 提交。

在仓库目录执行；所有 NPU 命令均须经队列：

```bash
source ../env.sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python tests/pypto_test/results/cann90_20260921/indexer_page_load_v1/compile.py
task-submit --device auto --max-time 1800 'bash tests/pypto_test/run_csa_validation.sh full_compare /tmp/csa_indexer_b4 /data/model/dsv4-flash-0731-dspark-w8a8 --batch 4 --history 131071'
task-submit --device auto --max-time 1800 'bash tests/pypto_test/run_csa_validation.sh full_compare /tmp/csa_indexer_b40 /data/model/dsv4-flash-0731-dspark-w8a8 --batch 40 --history 131073'
task-submit --device 8 --max-time 1800 'bash tests/pypto_test/run_csa_profiles.sh /tmp/csa_indexer_profile /data/model/dsv4-flash-0731-dspark-w8a8 --batch 40 --history 131073 --replays 3'
```

复核现有证据，无需上卡：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python tests/pypto_test/compare_dsv4_csa_profiles.py --root tests/pypto_test/results/cann90_20260921/indexer_page_load_v1/profile --swimlane-kernel-config tests/pypto_test/results/cann90_20260921/indexer_page_load_v1/profile/swimlane/kernel_config_source.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python tests/pypto_test/results/cann90_20260921/indexer_page_load_v1/summarize.py
```

离线比较初次发现三个 derived-zero weight-offset 记录来自 set，跨进程列表顺序变化。
比较工具现按唯一名称比较完整记录，同时拒绝重复名称；所有 shape、dtype、SHA256、
loaded_exact、source 字段仍参与比较。原始 metadata 未改动，证据见
`profile/weight_record_order.json`。修正仅影响离线报告，无需重跑 NPU。
43 份执行源码哈希保持捕获时版本；比较工具新版本独立记录在 `postprocessing_sources`。

本项使用 cann_recipe 参考层验证，不代表目标 ModelSlim 权重验收、完整模型性能或完整 P3
通过；先前连续 100 步精度问题不在本次优化范围。结果目录保留原始三次耗时与失败尝试。
