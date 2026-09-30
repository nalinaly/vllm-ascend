# TMR CSA/HCA atomic add 七档对照（2026-09-30）

用户要求在当前实现上比较 atomic_add 开启/关闭的性能，覆盖
128K/B4、B8、B16、B24，以及 8K/B16、B24、B32。
实验结果见 [RESULTS.md](RESULTS.md)，逐次样本及检查见 [evidence.json](evidence.json)。

七档已全部完成。开启相对关闭，按长短档 8:2 对各档均值变化百分比加权：
CSA +2.20%、HCA +2.23%、CSA→HCA 联合 +1.91%。四个 128K 档 CSA 均变慢，
联合图七档全部变慢；个别独立算子的小幅下降不构成统一开启的依据。
保持正式默认关闭，没有将实验性的 HCA atomic 放行合入生产。

14 组有限值、写入保护区和 metadata 检查通过，Top-K 均无结构非法。
atomic0 的三个图均与各自 eager 输出精确一致；atomic1 观察到重复执行的浮点变化。
开关两侧独立 CSA/HCA 的输出 max_abs 均为 0.015625；联合输出最高 0.03125。
每档 Top-K 有 1～6 行集合变化。本轮没有 token/DSpark 的整模型验证。
全部 42 组计时中最高 P95/median 为 1.0556；最大值原样保留，不删异常样本。
每组只有 20 次采样，不据此断言更低频的长尾不存在。

四个入口的无卡编译和三条设备队列任务均退出 0，任务编号及排除的驱动失败见
[tasks.json](tasks.json)。原始构建文件与输出张量只保存在 results，不提交重复大产物。

## 范围与配置

- 同一冻结源码，版本及冻结目录见 [source.json](source.json)。
- CANN 9.2.0-beta.2、NZ2、TMR、CSA 性能版、HCA 当前实现、TP1、S6、EPLB 关闭。
- 正式权重 `/data/model/DeepSeek-V4-Flash-0731-w8a8`，第 2 层 CSA、第 3 层 HCA。
  使用各自固定种子的合成输入/历史；不包含 MoE、16 卡或整模型测试。
- 两侧 Native 确定性 level=0、HCCL_DETERMINISTIC=false。
  PyPTO ring heap 每级 320 MiB，两侧相同，task window 使用运行时默认。
- 每档开关两侧使用同一张队列分配的卡，各用新进程，在导入算子之前固定开关。
- 图中只捕获根算子，区间覆盖 HC_pre+norm+attention+HC_post；compact metadata
  已准备并复用。权重加载、编译、初始化、状态恢复均在正式计时之外。
- 三种独立图：CSA、固定输入 HCA、CSA 输出直接接 HCA。联合耗时实际测量，
  不用两个独立均值相加代替。两层 attention 直接串接并非完整 transformer layer。
- 每图预热 5 次，正式采样 20 次，记录全部样本及 min/mean/max/P95/median；
  计时不用 profiler/DFX。每次调用之前 D2D 恢复同一 cache/state 初态并同步。

## 开关实际改变什么

该开关同时改变 QR/KV split-K、M 分工、GM 清零与写回方式，以及 KV 的 K 分块：

| 配置 | QR split-K | KV split-K | KV K tile | QR/KV 输出初始化 |
| --- | ---: | ---: | ---: | --- |
| atomic=0 | 1 | 1 | 512 | 独占写入，省去清零任务 |
| atomic=1 | 8 | 8 | 256 | 先清零，再做跨核 atomic add |

结果表示当前两套完整路径的性能，不能单独归因于硬件 atomic 指令。
Compressor、Indexer Score、Sparse Attention、O projection 没有因本实验新增 atomic。

HCA 的正式适配器/配置当前禁止 atomic=1，但其 QR/KV 调用共用上述 CSA 实现。
仅在冻结副本放开两个入口限制，补丁见 [hca_experiment.patch](hca_experiment.patch)。
没有修改生产限制、默认配置、kernel 算术或 cache 布局。

## 数值与有效性

每侧检查输出有限值、完整写入保护区和只读 metadata；固定 K 路径还要求
图/eager 输出精确一致。atomic 路径的图/eager 差异如实记录，不要求逐 bit 一致。
另保存三个图的输出及 Top-K，离线比较 atomic1 相对 atomic0 的 max_abs、RMSE
和 Top-K 集合变化。不保存重复的大体积全 cache 快照，不做额外 hash 校验。
本轮是开关性能实验，不把两侧的浮点差异自动判为精度失败，也不宣称 token/DSpark 验收。

## 复现

先在 `tests/pypto_test/results/atomic_add_seven_20260930/frozen_ops_pypto` 整包冻结
`vllm_ascend/ops/pypto`，应用上述两处 HCA 实验补丁。
CPU 编译使用 `run.sh <output> <0|1> --compile-only`，通过 `task-submit --no-device` 执行。
设备对照通过队列提交：

```bash
/data/server-toolkits/pto-task/app/task-submit --device auto --max-time 900 \
  "bash $PWD/tests/pypto_test/atomic_add_seven_20260930/run_pair.sh 131072 16 0"
```

末尾 `0` 为先关后开，`1` 为先开后关。队列自动附加的设备参数由驱动识别，
实际使用 `TASK_DEVICE` 并映射为进程内逻辑卡 0。
结果位于 `results/atomic_add_seven_20260930/h<history>_b<batch>/atomic<0|1>/`。
所有设备任务完成后，在公共环境运行 `collect.py` 汇总。
