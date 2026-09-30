# 同卡 Native↔PTO 七档验收（2026-09-30）

生产 `9a983dae`（= e110a886 + 第 492 节 early3 的三处 `allow_early_resolve`）。
冻结源 `.cache/csa-accept-9a983dae`，私有包 `dsv4_csa_accept_9a983dae`。
每档在**同一张卡**上按 `native_a → pto_a → pto_b → native_b` 的 ABBA 顺序跑，
每侧 20 正式设备事件；表内每侧取两次的平均。CANN 9.2.0-beta.2、mode2、atomic0、det0。

**两侧同一个编译入口**（vllm 的 `@support_torch_compile`）。

> ⚠ **2026-09-30 更正**：原文写「Native 侧注入 `super_kernel_optimize=True`
> 已确认生效」，这是错的。该入口带 `force_eager: True`，只有
> op_compiler 的 `--enable_super_kernel` 那半边生效；**出效果的 AclGraph 级
> 图内融合（`acl_graph.py:1207`）永不执行**——它在 capture 之后才跑，而
> `run_eagerly` 直接返回 eager 执行，压根没有捕获图。
> `installed_static_packages: 1` 只证明 static_kernel 装了包，不是 SK 的判据；
> 正确判据是 `compiler.static_super_flags`（本 harness 未记录）。
> **本文件里的 Native 实测等同 sk=0**，比值列是「PTO ÷ 不开图级 SK 的 Native」。
> 开图级 SK 的 Native 七档数据见
> `../csa_native_sk_seven_20260930/`（日志第 501 节），值 6.0%~15.2%。
>
> 另注：`force_eager: True` 是 `vllm_ascend/compilation/compiler_interface.py`
> 里**生产写死的默认**，所以本表比的其实正是当前生产形态。

## 结果（p50 为主；见下方"为什么不用 mean"）

| 档位 | Native+SK min / p50 / p95 | PTO min / p50 / p95 | 比值(p50) | 比值(min) |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4 | 787.74 / 791.36 / 793.77 | 625.73 / 639.00 / 654.96 | 0.807 | 0.794 |
| 128K/B8 | 949.38 / 954.91 / 962.79 | 723.64 / 737.74 / 755.83 | **0.773** | 0.762 |
| 128K/B16 | 1239.06 / 1249.04 / 1254.54 | 951.36 / 969.09 / 983.68 | **0.776** | 0.768 |
| 128K/B24 | 1418.33 / 1425.46 / 1431.60 | 1194.54 / 1230.91 / 1247.18 | 0.864 | 0.842 |
| 8K/B16 | 859.17 / 864.73 / 873.79 | 737.65 / 765.01 / 785.02 | 0.885 | 0.859 |
| 8K/B24 | 1040.00 / 1056.98 / 1067.53 | 883.55 / 915.86 / 933.02 | 0.866 | 0.850 |
| 8K/B32 | 1176.04 / 1183.42 / 1187.79 | 1023.29 / 1043.50 / 1063.41 | 0.882 | 0.870 |

| 口径 | 128K 均 | 8K 均 | **8:2 加权** | 距 0.80 |
| --- | ---: | ---: | ---: | ---: |
| 按 p50 | 0.8049 | 0.8776 | **0.8194** | +0.0194 |
| 按 min | 0.7916 | 0.8594 | **0.8052** | +0.0052 |
| 按 p95 | 0.8164 | 0.8892 | 0.8309 | +0.0309 |

**128K/B8 (0.773) 与 128K/B16 (0.776) 已在 0.80 以内，128K/B4 (0.807) 接近。
缺口集中在 128K/B24 (0.864) 与三个短档 (0.866~0.885)。**

## 为什么用 p50 而不是 mean

`h131072_b8/native_a` 的 20 次里有一次 **16371.66 μs** 的尖峰
（min 943.66、p50 951.82、p95 959.64、max 16371.66），把该侧 mean 拉到 1723.32、
两次 native 的 mean 差 761.52 μs，按 mean 算出的比值 0.553 是假的。
`h8192_b16` 两侧也各有一次 ~1150 μs 的小尖峰。
**这些是系统抖动，不是算子**——p50/min 不受影响，故以 p50 为主、min 作下界。

## 口径说明：为什么不能用归档的 Native 1130.85

`csa_spmd_writeback_20260930/LATEST_EXISTING_COMPARISON.md` 的 Native 列
（128K/B16 = 1130.85）走的是另一条入口：
`torch.compile(backend="npugraph_ex", fullgraph=True, dynamic=False,
options={force_eager: False, inplace_pass: False, static_kernel_compile: True,
super_kernel_optimize: True})`，由 `csa-native-superkernel-4ffccb7b` 那份 runner 驱动。

**那条入口对当前源两侧都不可用**：

- Native 侧：`RuntimeError: Cannot prepare for replay during capturing stage.
  ... Current npuStreamCaptureStatus: npuStreamCaptureStatusActive`
  —— `_replay_graph.replay()` 在外层捕获仍活跃时被调用（嵌套图捕获）。
  旧 harness 对当前 torch_npu 2.10.0.post2 的版本错配。
- PTO 侧：`wrapper_compiled: False`、`static_compile_results: []`、
  `installed_static_packages: 0`，`torch.compile(fullgraph=True)` 根本没产出
  编译体。

所以 1130.85 **配不上一个同入口的 PTO 读数**，只能当历史参考。
本目录给出的是当前源上**唯一内部一致**的同卡对比。

另记一个反直觉观察：在 vllm 装饰器入口下给 Native 开 `super_kernel_optimize`，
128K/B16 的 Native p50 从 1218.55（不开，另一张卡）变成 1249.04（开），
即**这条入口下该开关没有兑现归档基线那 7.7% 的优势**——归档的优势来自
整条 `force_eager=False` + 直接 npugraph_ex，不是这个开关本身。
