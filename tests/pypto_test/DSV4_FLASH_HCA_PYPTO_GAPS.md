# DSV4 HCA：PTO 与手写 Native 的差距落在哪里——给 PyPTO 的四个问题

> 面向 PyPTO 团队。数据全部来自 `DSV4_FLASH_HCA_OPTIMIZATION_LOG.md`
> 第 84–95 节（2026-09-29 一轮），口径是单卡 128K/B16、同卡 ABBA、
> 每侧 ≥6 样本、纯 device span。

## 0. 结论先说

HCA 的 PTO 算子在七档上是 Native 的 **1.117 倍**（目标 0.80）。这一轮测了
**十个候选，没有一个为负**；此前累计 18 个候选只落地过一次 −47 μs。
算子侧的调参空间与 PyPTO 两层配置面（`PassContext`、`pypto.torch.init`）
已实测穷举。**剩下的差距不在算子写法上，而在四个 PyPTO 层面的问题上。**

## 1. 同样的标量预算，Native 推进 1.8 倍的 MAC

`torch_npu.profiler.AiCMetrics(PipeUtilization)`，同一次运行的两侧：

| kernel | 墙钟 | mac | scalar | mte1 | mte2 | fixpipe |
| --- | --- | --- | --- | --- | --- | --- |
| PTO `aicore_kernel_mode_0` | 663.2 | **14.9%** | 59.2% | 17.2% | 29.9% | 12.0% |
| Native `SparseAttnSharedkv`（手写 AscendC） | 186.1 | **26.7%** | 59.4% | 33.0% | 91.4% | 90.8% |

**两侧 scalar 占比几乎相同（59.2% / 59.4%），而 mac/scalar 是 0.25 对 0.45。**
AIC 平均单核有 387 μs 花在标量上（占 654 μs 墙钟的 59.2%），
而它均摊在「134 行循环体 × 144 次执行」上、**没有单一热点**。

也就是说：PTO 的 cube 不是在等搬运（mte2 仅 29.9%，Native 才是 91.4%），
而是在等标量算完地址、判完边界、对完 flag。这部分由 PyPTO 的代码生成决定。

## 2. profiler 分不出"标量在执行"与"标量在等待"——这是最卡的一点

`aic_scalar_ratio` 是"标量单元非空闲的时间占比"，两者合在一起。
把 `AiCMetrics` 全部模式试过：唯一与停顿相关的 `ResourceConflictRatio`
给出的三列全是 `aiv_` 前缀（`aiv_vec_bankgroup/bank/resc_cflt_ratio`），
**没有任何 AIC 列**；而 AIV 的冲突率本身只有 3.7%/1.8%，不是瓶颈。

后果是**优化方向无法判定**，只能靠改一版测一版。而本轮实测的测量分辨力是：
同一张卡 base 的 6 样本极差可达 58 μs、σ 约 25，**10 μs 以下的效应
拿同卡 ABBA 只会得到随机符号**（`cmp_wgate` 四轮 −8.0/−13/−10.5/+2.75，
`aicpu_thread_num=4` 三样本 −11.0 补测后 +5.0）。

**请求**：给 AIC 一条能区分标量执行与标量等待的计数通道（PMU 事件或
incore 模拟器的周期级导出），否则 μs 量级的标量优化在这条链上不可验证。

两个反例说明"看起来合理的代理指标"都不可用：

- 数生成的 AscendC 里的同步原语：`qk_manual` 把 `pipe_barrier` 从 11 降到 4
  （−64%），耗时 **+12.88 μs**。同步计数与耗时不相关。
- 删除死代码：`_hc_pre_partials` 里三行 `pl.read(hc_scale,[i])` 读了从不使用，
  删掉后 **+21.88 μs**（6 样本，区间完全分离）。

## 3. NZ 与 Vector 核 L2 预热互斥，五条绕法全部受阻

HCA 有三张 cube matmul 的 B 操作数权重（`wkv`、`cmp_wkv`、`cmp_wgate`）
为了 L2 预热而留在 ND：冷 L2 下单个 AIC 核读权重只有约 24 GB/s，
所以用 8 个空闲 Vector 核按块读一遍把它们带进 L2。实测这份预热值
**+25.75 μs**（删掉两张的预热，6 样本）。

把它们改成 NZ 后预热失效，PyPTO 明确拒绝：

```
ValueError: NZ layout currently supports only matmul operand loads
(target_memory=pl.Mem.Mat), got Vec.
```

`BlockNzTensorViews` 的文档留了出路——"覆盖全部元素的 rank-1 view 与 layout
无关……正好是 `prefetch.async_prefetch` 对源张量的要求"——但五条路都走不通：

| 尝试 | 结果 |
| --- | --- |
| 直接 2D 窗口读 | ✗ 上面那条布局检查 |
| 调用点内联 `pl.reshape(w,[1,N])` 传给预热函数 | ✗ `@pl.jit: missing inferred tensor metadata` |
| 先落成具名变量再传 | ✗ 同上（元数据推断只认直接传入的根参数） |
| 在被调函数体内 reshape 成 rank-1 `[N]` | 过了布局检查，✗ **codegen 无 1D GM→UB 搬运** |
| 同一存储再绑一个不带 layout 的 ND 别名参数（预热不消费数值） | ✗ `Parameter requires base format NCHW (0) or ND (2)` |

**请求**：支持 NZ 张量的整块 rank-1 读入 Vec（文档规则已经写好，缺 codegen
那条 1D GM→UB 搬运），或提供一条"只为把字节带进 L2"的预热原语。
否则 NZ 与 L2 预热只能二选一，而实测两者量级相当（NZ 自身约 −23 μs、
预热约 +12.9 μs／张），互相抵消到测不出来。

## 4. `runtime=host_build_graph` 与 torch NPUGraph 不兼容

泳道显示 device span 由 AICPU 建图而非计算决定：
`simpler_aicpu_kernel_exec` **683.1 μs** vs 计算核 **663.2 μs**，
且晚 18 μs 收尾；31 个用例里 AICPU 一律占 span 的 98.5%+。

`host_build_graph` 的文档写它 "required for Graph Execution"，看起来正对症。
三个阻塞都能零接口改动解开（`runtime` 要在 `pypto.torch.init` 传而非
`PassContext`；`hc_pre_norm` 的 Host 层 `pl.read` 可下移进 spmd；
`_hc_pre_partials` 那组是死代码）。但编译通过后：

| 侧 | span |
| --- | --- |
| Native | 548.8 / 534.5 |
| PTO（HBG） | **90 923.5 / 94 586.0** |

**慢约 157 倍。** 原因是那里的 "Graph Execution" 指 PyPTO 自己的 graph 模式
（`pl.graph` / `@pl.jit.graph`），而本项目是把 PTO 调用捕获进 **torch_npu 的
NPUGraph** 再重放。HBG 在这条路径上拿不到 host 端图，退化成每次调用重建整图。

**请求**：(a) 文档里明确 `host_build_graph` 不适用于 torch NPUGraph 捕获路径，
或在该组合下直接报错而不是静默退化 157 倍；(b) 若 AICPU 建图开销
（683 μs 里相对计算的那 20~29 μs 尾巴）有别的减法，请指路——
它是唯一被数据证明决定 span 的一项。

## 5. 附：本轮十个候选的完整数据（供判断是否已是局部最优）

| 类别 | 候选 | Δ vs base |
| --- | --- | --- |
| 权重布局 | `wkv` → NZ | +14.75 |
| 权重布局 | `cmp_wkv` → NZ | +20.25 |
| 权重布局 | `cmp_wgate` → NZ | 四轮 −8.0/−13/−10.5/+2.75（符号不稳） |
| 核间流水 | `gather_full`（gather 块数 32→64） | +19.00 |
| 核间流水 | `gather_pipe`（内层加软流水） | +10.25 |
| 核内 | `qk_manual`（QK 改手工 split-K） | +12.88 |
| 运行时 | `aicpu_thread_num=2` | 崩溃（`npuSynchronizeDevice`） |
| 运行时 | `aicpu_thread_num=4` | −11.00 → 补测 **+5.00** |
| 运行时 | `aicpu_thread_num=5` | 初始化失败（code −1000） |
| 运行时 | `runtime=host_build_graph` | 慢 157 倍 |
| 代码清理 | 删除三行未使用的 `pl.read` | **+21.88** |

十个候选无一为负，范围 +10 μs 到 +157 倍。连删除未使用的代码都要付 21.88 μs，
说明这一版算子处在一个**被扰动即退化**的局部最优上。
若 PyPTO 侧的四个问题都不动，1.117 大概就是当前分解与当前 codegen 的下限。
