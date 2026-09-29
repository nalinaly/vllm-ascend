> **长期保留的 HCA 性能优化过程日志；本文件为调测信息的唯一权威记录。**
> 与 CSA 侧的 [DSV4_FLASH_CSA_VALIDATION_LOG.md](../../../vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/DSV4_FLASH_CSA_VALIDATION_LOG.md) 对等。
> 接入合同、剩余事项与交接状态以 [DSV4_FLASH_HCA_HANDOFF.md](DSV4_FLASH_HCA_HANDOFF.md) 为准；
> 数据文件与源码快照在 [results/hca_optimization_20260928/](results/hca_optimization_20260928/)。
> 每节记录：修改原因、改了什么、测试配置、实测结果、结论边界、下一步。
> 各节的"当前"、PASS 与权限只对该节的日期、源码与工具链成立，不自动沿用到后续节。
> **尤其注意 CANN 版本分界（见第 9 节）与计时口径（见第 10 节）。**

# DeepSeek-V4 Flash HCA 单层性能优化过程记录

## 0. 当前有效结论索引（先读这一节）

本文件是追加式过程日志，**中间有多条结论被后续实测否定**。照着作废的结论行动会走弯路，
因此先在这里列清"现在还成立什么"。下面每条都注明依据节号。

### 0.1 历史性能证据索引（最新Native对照见第110节，核内采用见第111/115节，候选筛选见第112–114节）

2026-09-30新同卡128K/B16部署口径：Native/PTO a7a9f314 mean为544.35/614.94μs，
PTO仍慢12.97%，20%目标未达到。第111节flat Gather和第115节M4/D512仅已证明各自核内收益，
尚无改动后的新七档或整机验收；以下历史比值不能替代该结论。

| 结论 | 依据 | 证据强度 |
| --- | --- | --- |
| `wo_a` 的布局必须跟随 Native 的实际存法（9.0 是 ND、9.2 是 NZ），由 `_native_keeps_3d_bf16_as_nz()` 探测 | 第 12 节 | **强**：proj_a 核时间 2624→1408 μs（−46%），端到端 128K/B16 −41 μs，约 3 倍噪声底 |
| 冻结候选快照时，共享的 `nz_mode.py` / `native_adapter.py` 必须跟生产版同步 | 第 12 节 | 强：不同步会静默 recast，量到的不是生产路径 |
| 七档验收比值均值 0.965～0.972，距 0.80×Native 差 69～185 μs，**未达标** | 第 22、27 节 | 强 |
| 该轮AIC核时间 ÷ 24 = 383 μs，整层625 μs，目标538 μs；这是资源下界，不能单独证明唯一瓶颈 | 第29节 | 下界估算，不是可实现性能保证 |

### 0.2 已测候选的历史结论（仅对对应配置有效）

| 方向 | 结论 | 依据 |
| --- | --- | --- |
| proj_a 的行块 M 分档 | 无效，cube 本就按 `valid_shape` 只算有效行 | 第 11 节 v13 |
| proj_a 的列块 N128→N256 | NZ 下无带宽可换，只剩填充率损失 | 第 19 节 v19 |
| proj_a 增行块 + wo_a 走 L2 | 明显更差 | 第 11 节 v15/v16 |
| 去掉 O 投影的宽扇入 dummy | 更差：48 块消费者带 dummy 时每块只跟踪 1 条依赖边，去掉后是 8 条 | 第 24 节 v23 |
| 去掉压缩器的 `late_dep` | 更差：它是有意的 cube 错峰 | 第 26 节 v24/v25 |
| 串联 dequant 与 `cmp_work_gather` | 无效：并行改接续，合计跨度不变 | 第 31 节 v27 |
| 配平并发任务块数（dequant/gather 各 24） | **端到端更差 18.4 μs**，尽管泳道全面改善 | 第 33 节 v29 |
| 去掉 AICPU 的 dummy 调度依赖改直连 | 无效且方向相反：dummy 每层仅 2～4 个、3～10 μs，远在 ±13 μs 噪声底以下；直连把 N+M 条依赖边变成 N×M 条 | 第 34 节 |
| 拆 gather 对压缩器的栅栏（v30/v31） | 加权 −7.1 μs，在 ±13 μs 噪声底内；8K 对照档同代码测出 +9.54 μs | 第 37.1 节 |
| **细分块数让任务占满核**（`CACHE_WORKERS`/`ROPE_CS_T_TILE`/`VALID_TOKEN_TILE`/`MM_ROWS`） | **加权 +24.4 μs，三档全为正**；连带说明块数两个方向都不是杠杆 | 第 42.1 节 |
| 给小任务补 `allow_early_resolve=True` | 加权 −1.07 μs，等于没有；大块任务才有效（v21） | 第 42.2 节 |
| **proj_a 的 tiling（M/N/K 全部维度）** | N 只能 128（NZ 粒度 vs L0C 双缓冲两侧夹住）、K 只能 256（L0B 每片 ≤64 KiB）、M 无差别（cube 只算有效行）；五个候选全否 | 第 55、57 节 |
| 把 cmp gather 融进 attention（CSA 式） | 不可行：worker 的 token 跨请求，同请求 6 个 token 不共享 KV 载入，融合会把 gather 做约 6 遍 | 第 57.3 节 |
| HCA 自己的 ring arena 下限探测 | 不需要，`wo_a` 布局修正后 128K/B24 显存缺口已消失 | 第 13 节 |

### 0.3 三条必须遵守的计时口径（每条都是踩坑换来的）

1. **`run_hca_reference_pair.sh` 的"基线→候选"差值不能当收益**：它先跑基线后跑候选，
   候选总在更热的卡上测。七档里可信的只有同一轮内测出的 PTO/Native 比值。见第 10 节。
2. **ABBA 的净收益要扣掉同轮 Native 漂移**，且**噪声底为 ±13 μs**（标准差 9.05，
   零假设实验见第 28.1 节）。单次 ABBA 只能分辨约 25 μs 以上的效应。
3. **泳道的跨度／起跑摊开不是端到端时间的可靠代理**，只能用于解释现象与定位方向；
   核时间与单块均值可以作判据。见第 33 节。这一条纠正了第 28.2 节。

### 0.4 已作废的表述（勿引用）

- 第 15、28.2 节的"阶段边界空档 80～113 μs"：那是按功能组起止时刻相减算的，组间重叠，
  真实全空窗只有 64 μs（含 31 μs 启动）。见第 29 节。
- 第 31 节的"664 块 × 0.33 μs = 221 μs 派发开销"：每块派发不是常数（N=48 时 1.12、
  N=32 时 0.33）。见第 32 节。
- 第 32 节的"索要满核有害"：反例是 `attention_aic` 与 `oproj_hc_post` 同为满核而摊开仅
  2.8／2.7 μs。见第 33 节。
- 第 20、23、24、26 节中一切 ±20 μs 以内的净收益数字：落在噪声底内，仅作历史记录。

### 0.5 四道 PyPTO 写法限制

1. `pl.array` 不能跨 `@pl.jit.inline` 边界返回（codegen 报 `ArrayType`），但**可以返回它的元素**。
2. `return` 里不能用列表推导（`Unsupported expression type: ListComp`），同样的推导式作为
   `task_dummy(deps=...)` 的实参却允许。
3. `pl.spmd(deps=...)` 的每一项必须是 TaskId 变量或 `arr[i]`，元组下标表达式被拒。
4. `pl.range` 追踪循环内创建的 TASK_ID 在循环外不可见；用一元 TASK_ID 数组承载可绕开
   （本文件 `q_proj_qr` 早有此惯用法）。

### 0.6 旧阶段提出的方向（2026-09-30起不作为范围穷尽判断）

1. 改 `simpler/` 的派发/调度实现（按范围界定在范围内，但它是 CSA 共用组件，
   且与上游 `SUBMIT_BY_CLUSTER.md` 的 MixedTask 设计重叠）。注意：改完仍要用正式计时验证，
   而分辨力只有 ±13 μs。
2. 优化版整机 TP1×DP/EP16 token 验证（交接文档剩余事项第 1 项，需 16 卡档期）。

算子侧已测候选见0.2。它们只排除各自配置，不能证明所有数据复用、流水或调度方案已穷尽；新阶段见第99节。


## 1. 目标、验收口径与执行顺序（2026-09-28）

目标由交接文档定义：每个单卡档位的 PTO P50 ≤ 同卡 Native P50 × 0.80，即领先 Native 20%。
流程严格单卡先行，再整机；整机验收要求输出 token 逐 bit 相同。

范围（用户逐步明确）：

- HCA 算子内的**任务调度与 incore task 核内耗时都在范围内**，不同于 CSA 只看 incore task。
- 本仓的 `pypto/`、`simpler/` 框架代码也在范围内。
- DLPack 的 NZ 别名方案不做。
- NZ 相关做法先查 CSA 会话在其验证日志里的既有结论，不自己另起方案。
- kernel 模式每次调用约 80 μs 的固定开销属公共优化，本会话不处理。
- **工作顺序：先逐个对照 incore task 与 Native 同功能算子，再看 task 之间的调度空间。**
- 后续候选按 **128K:8K = 8:2** 加权判断；整体有效就保留，严重顾此失彼时按档位分支。

档位：原七档为 128K×B4/B8/B16 + 8K×B16/B24/B32/B40；第 13 节起新增 128K/B24，
整机验证通过后退役 8K/B16。

## 2. wo_a 的公平性修正：先改 ND（CANN 9.0 口径）

发现 HCA 把 `wo_a` 声明为 NZ，而 Native 在 CANN 9.0 上把这个三维 BF16 权重留在 ND。
这既逼出每层 64 MiB 的私有副本，也等于给了 PTO 一个 Native 在同配置下拿不到的布局，
违反 CSA 第 144 节的公平对比规则。改为 `WO_A_WEIGHT_LAYOUT`（当时解析为 ND），
`decode_hca.py` 1 处 + `o_proj_hc_post.py` 2 处。`source_prod_woa` 成为新的公平基线，
此前的七档数字只作历史保留。**注意这一节的结论在 CANN 9.2 上反转，见第 12 节。**

## 3. 冷热 L2 归因与两个新的泳道采集模式

正式计时是交替跑 Native/PTO，PTO 的权重是冷的；而 DFX 泳道是连续重放、权重是热的。
实测冷跨度 604～636 μs、热跨度 476～489 μs，`proj_a` 每块 42.9 vs 14.9 μs，差异显著。
为在正确的热态下取证，给 `dsv4_hca_performance.py` 的 `capture_swimlane` 加了两个模式，
并在 `dsv4_hca_single_layer.py` 暴露为开关：

- `--swimlane-cold-l2`：每个窗口前用 512 MiB 写入冲掉 L2。
- `--swimlane-after-native`：每个窗口前先跑一次 Native，复现正式计时的热态。

## 4. kernel 模式启动固定开销探针（结论：不在本会话范围）

新增 `dsv4_pto_launch_floor.py`，用与单层计时相同的方式（torch custom op 捕获进
NPUGraph、图外 NPU Event）测极小 kernel 的重放开销：1 任务×1 块 = 79.7 μs，
32 个串联单块任务 = 151.6 μs，8×48 并行 = 171.9 μs，75×8 = 476 μs；调整 ring 配置
不改变这个底噪。踩到两个坑：`@pl.jit` 编译时重读源文件，闭包生成的 kernel 会报
`OSError: could not locate function definition`，改为模块级函数加 `KERNELS` 表；
单块收尾任务不能用 `pl.spmd(1)` 而要用 `pl.at(level=pl.Level.CORE_GROUP)`。
**用户明确这属公共优化，本会话不再推进。** 该 ~80 μs 计入 PTO 的实测 P50，
是后面判断"目标可达性"时必须计入的固定项。

## 5. incore 逐功能对照方法与首轮结果（CANN 9.0，v8）

新增 `summarize_hca_incore.py`：Native 取同轮计时后的 PyTorch profiler
（`kernel_details.csv`，基本单流串行，算子时长即该功能耗时），PTO 取"Native 之后"
口径的 DFX 泳道（只取 pid 4 的 Worker View）。PTO 任务之间有重叠，所以给出
每个功能组的跨度、单块核内均值/最大与核时间总和，**不把跨度相加当作整层时长**。

首轮（v8，8K/B16）多数 incore task 已快于 Native；关键链上落后的是
Q_B（43.6→56.3/56.9 μs）与 O 投影（138.4→150.6/152.1 μs），另有约 34～52 μs 的
PTO 独有准备任务与约 116 μs 的阶段边界空档。

## 6. 候选 v1～v9 的取舍（CANN 9.0）

- 保留：v6（残差加宽与 RMS 合并，移植 CSA d1f170ff）、v7（v6 + CSA 56186de9 的
  `qkv_proj_rope.py`）、v8（v7 去掉 wo_a 预热）。
- 否定：v2/v5（AIV 预热 ND 权重，无净收益）、v3a/v3b/v3c（proj_a stage=3/4 超 L0 容量：
  `Mat buffer usage (589824) exceeds 524288`、`Right buffer usage (131072) exceeds 65536`，
  与 CSA 第 120 节已记录的否定结论一致）、v9（proj_a N256，无收益）。
- 期间踩到：`weight_warm.py` 复用同一变量名承载不同 tile 形状导致
  `ParserTypeError: Cannot reassign 'tile' with a different type`，改为每个张量独立命名；
  NZ 权重不能被 AIV 预取（`NZ layout currently supports only matmul operand loads`），
  预热只能覆盖 ND 权重。

## 7. v10（Q_B 重分块）与 v11（准备任务延后）

- **v10**：`_q_proj_q_matmul_nz` 改为在每个 N256 列块内按 K256 双缓冲、M128 行块覆盖
  有效行，首段用 `pl.matmul` 生成携带有效行 compact 形态的累加器。
  起因是直接对 `create_tensor` 累加器做 `matmul_acc` 且只有部分有效行时会失败
  （`AccCompactValid`）。INT32 精确，数值不变。
- **v11**：纯调度。`hca_raw_valid` 与 `hca_inverse_rope_sign` 改为依赖 SWA 写回完成，
  从而在 Q_B 窗口内执行；KV 预热块数 24→8，减少与 mHC pre 阶段边界同时到来的完成事件。

两者相对 v8 的 ABBA（CANN 9.2）：v10 使 8K 相对 Native 从 +5.87% 降到 +2.82%、
128K 从 +0.82% 到 +0.56%；v11 使 8K 从 +5.72% 到 +4.33%、128K 从 −0.31% 到 −0.82%。

## 8. v12 = v10 + v11

两者改动文件不相交（v10 在 `deepseek_v4_flash_dspark/q_projection.py` 与
`deepseek_v4_flash_dspark_perf/qkv_proj_rope.py`；v11 在
`deepseek_v4_flash_hca/decode_sparse_attn_hca.py` 与 `weight_warm.py`），
`source_combo_v12` 由 v8 叠这四个文件生成并逐文件 `cmp` 校验来源。
按第 10 节的口径，v12 相对 v8 的净收益为 8K −12.2 μs、128K −16.1 μs（约 2%）。

## 9. ⚠ 版本分界：公共入口在 21:59 切到 CANN 9.2.0-beta.2

另一个会话在 2026-09-28 **21:59:13** 把公共入口 `env-dsv4-0251rc1.sh` 改为 source
`cann-9.2.0-beta.2/set_env.sh`（原 `/usr/local/Ascend/cann-9.0.0`）。
`run_hca_single_layer.sh` 会 source 这个入口，因此**任务按启动时刻继承版本**：

| 版本 | 属于该版本的取证 |
| --- | --- |
| CANN 9.0.0 | 原七档基线、`source_prod_woa` 对照、v1–v9、`incore_v8_*`、launch floor 探针、交付的 `hca_h131072_b16_profiles/` |
| CANN 9.2.0-beta.2 | `abba_v8_v1x_*`、`seven_cann92_*`、`incore_cann92_*`、`metadata_replay_cann92/` |

核对方式是看结果目录 `build_output`／日志里实际出现的 CANN 路径，
**不能看当前 shell 的 `ASCEND_HOME_PATH`**——已在运行的 shell 仍是旧值，我据此答错过一次。

同一份 v8 代码跨版本：8K Native/PTO 551.6/597.5 → 558.3/597.5；
128K 684.7/685.8 → 703.4/708.4 μs。两侧都略慢、比值几乎不动，
**0.8×Native 的目标在两个版本上一样远**。同一次 ABBA 内部的相对结论不受影响
（版本是共同项），但绝对基线与 incore 对照表必须在 9.2 上重取。

## 10. 计时口径纠正（影响此前所有"基线→候选"差值的读法）

两个必须遵守的规则：

1. **`run_hca_reference_pair.sh`（七档使用）先跑基线、后跑候选两个独立进程，
   候选总在更热的卡上测量，该差值系统性偏向候选，不能当收益。** 两次七档之间也不可比：
   同一份 Native 代码在 8K/B16 上一次 583.9、一次 544.8，差 39 μs。
   七档里可信的只有**同一轮内**测出的 PTO/Native 比值。
2. **ABBA 的净收益要再扣掉 Native 的同期漂移。** Native 在四轮里跑的是同一份代码，
   其"变化量"就是该卡的漂移。净收益 =（PTO 候选轮均值 − PTO 基线轮均值）−
   （Native 候选轮均值 − Native 基线轮均值）。只给比值差会把漂移算成收益。

## 11. 对照 CSA 的分档 tiling：v13～v16 全部否定

用户指出"参考 CSA 的算子，可以对不同的 shape、档位实施不同的核内 tiling 策略"。
查证 CSA 验证日志第 175 节 v3"O projection 自适应分块"：O-A 按 token 数选 N128/N256；
O-B 选 M32/M96/M128、N256；大 token 档位 NZ O-B 权重完整 K 常驻、小档位 K256 流水；
并且明确**保留 Native 的 `[G,K,N]`／`[G*K,N]` 描述与原数据指针、不做重排**。
这些在 HCA 侧已经是 `decode_o_proj_tp1` 的逐字拷贝，`weight_resident` 分支也在。
**CSA 唯一没覆盖的维度是 proj_a 的行块 M。**

| 候选 | 改动 | 8K 净收益 | 128K 净收益 | 结论 |
| --- | --- | ---: | ---: | --- |
| v13 | proj_a 行块按档位贴合（32/48/96/80×2/96×2/120×2），行块数不变 | +5.51 | +0.11 | 否：cube 本就按 `valid_shape` 只算有效行 |
| v14 | proj_a 列块 N128→N256 下放到 ≤96 档 | +7.13 | −7.05 | 当时否；按 8:2 加权为 −4.21，需在 v17 的 NZ 路径上重测 |
| v15 | N256 + 3 行块（96 块/4 满波）+ 列块最外 + wo_a 走 L2 | +44.89 | +44.56 | 否，明显更差 |
| v16 | 同 v15 但保留 `CachePolicy.BYPASS` | +52.50 | +39.28 | 否，明显更差 |

v14 的 incore 给出了当时的定位：proj_a 从 64 块×41.0 μs（核时间 2624）变为
32 块×64.0 μs（核时间 2048），核内效率追平 Native 的 86 μs，但块数腰斩使 24 AIC 填充率
从 89% 降到 67%，收益被抵消。v15 的 3 行块把核时间推到 3398 μs——重复读权重三倍，
L2 没接住。**这一整条线索建立在错误前提上，见下一节。**

## 12. 根因：wo_a 的布局必须跟随 Native 的实际存法（v17）

用户要求**性能测试优先 vllm_ascend 的 nz_mode=2**。查证后发现使第 11 节全部失效的前提：

候选快照用 `--operator-source` 重指 `vllm_ascend.ops.pypto.__path__`，连带冻结了共享的
`deepseek_v4_flash_dspark/nz_mode.py`，里面是第 2 节留下的 `WO_A_WEIGHT_LAYOUT = None`。
而 `wo_a` 在 Native 侧存 ND 还是 NZ 取决于当前 CANN 的 `npu_format_cast` 支不支持
三维 BF16：**9.0.0 不支持（Native 留 ND），9.2.0 支持（Native 是 NZ）**。
于是在 9.2 上写死 ND 反过来逼出私有副本，每层 78.18 MiB、21 个 ratio-4 层合计
**1.603 GiB**（CSA 侧 `results/mem_128k_b24_20260928/ANALYSIS.md` 第 19 节有同源数据）。
生产版已改为 `_native_keeps_3d_bf16_as_nz()` 导入期用 4 KiB 三维 BF16 张量探测，
并在格式不匹配时打 `PTO_CSA_WEIGHT_RECAST` 告警（此前完全静默）。

**结论：我在 9.2 上量的 proj_a 一直是那份 recast 副本，不是 nz_mode=2 的生产路径。**
"ND 列切片跨步读只有 519 GB/s"对副本成立，但生产路径上 NZ 的列切片本身连续。
教训与既有约束"对照 Native 路径、不要自己发明判据"一致：**布局不是可以选择的常量。**
操作层面的推论：**冻结快照做性能对照时，共享的 `nz_mode.py` 与 `native_adapter.py`
必须跟当前生产版同步。**

`source_combo_v17` = v12 + 生产版共享文件（只刷新 1f7bad79 / 715e504d / 1a88c7b4 动过的
`ops/pypto` 文件：`dspark/{nz_mode,native_adapter,decode_o_proj,decode_csa}.py`、
`dspark_perf/{nz_mode,decode_csa}.py`、`variant.py`），唯一实质变量是 wo_a 布局跟随探测。

ABBA 净收益：8K/B16 **−6.98**、128K/B16 **−41.07**、128K/B24 **−15.26 μs**。
proj_a 的 incore 变化（均为 64 块）：

| 版本 | 单块均值 μs | 核时间 μs | 核时间/24 | 跨度 μs | Native 同功能 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v12（recast 出的 ND）8K | 41.0 | 2624 | 109.3 | 129.2 | 86.2 |
| v17（NZ）8K | 22.0 | 1408 | 58.7 | 75.3 | 70.1 |
| v17（NZ）128K | 22.2 | 1421 | 59.2 | 80.4 | 82.8 |

核时间降 46%；整个 O 投影组由落后 +23.5 μs 转为 8K −1.9、**128K −18.0（领先 Native）**。
整层 DFX 跨度 8K 从 563.1 降到 498.6 μs，128K 为 629.0 μs。
v17 所有运行的 `PTO_CSA_WEIGHT_RECAST` 告警为 **0**。

## 13. 八档验收表与新档位 128K/B24（CANN 9.2，v17）

对 `source_prod_woa` 的逐 bit 门禁八档全 PASS。

| 档位 | Native P50 | 基线 PTO | v17 PTO | v17/Native | 目标 0.8× | 还差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128K B4 | 406.5 | 472.7 | 403.5 | 0.993 | 325.2 | +78.3 |
| 128K B8 | 541.8 | 581.6 | 503.9 | **0.930** | 433.4 | +70.5 |
| 128K B16 | 692.7 | 750.1 | 654.9 | 0.946 | 554.1 | +100.8 |
| 128K B24（新） | 859.1 | 887.0 | 809.8 | 0.943 | 687.3 | +122.5 |
| 8K B16 | 538.1 | 636.7 | 528.7 | 0.983 | 430.5 | +98.3 |
| 8K B24 | 680.1 | 764.9 | 653.6 | 0.961 | 544.1 | +109.5 |
| 8K B32 | 761.9 | 822.4 | 735.4 | 0.965 | 609.6 | +125.8 |
| 8K B40 | 823.4 | 945.7 | 823.2 | 1.000 | 658.7 | +164.5 |

比值均值 0.965（v12 时 1.008），八档全部 ≤1.0，距 0.80× 仍差 70～165 μs。

**128K/B24 显存问题已随第 12 节的布局修正消失**：PTO 此前在该档位差 Native 1.63 GiB，
与 recast 的 1.603 GiB 几乎完全吻合。因此 CSA 侧为该档位实测的 ring arena 配方
（`heap_mb=[256,128,256,32]` + `task_window=4096`，省 0.574 GiB）对 HCA **不需要**，
不做 HCA 自己的 ring 下限探测。

## 14. 单层同地址 metadata A→B→A 重放 PASS

`dsv4_hca_metadata_replay.py` + `run_hca_metadata_replay.sh`，数据在
`results/hca_optimization_20260928/metadata_replay_cann92/`。B4、history A=124 / B=8190、
统一页表宽度、B 的页表行反序（为此给 `dsv4_csa_single_layer.py` 的 `make_fixture`
加了 `table_history` 与 `reverse_pages` 两个参数，使 A/B 能共用形状与地址）。
54 个 metadata 张量叶子中 23 个在 A/B 间指针与内容都不同（`block_table`、
`input_positions`、`qli_metadata`、`sas_metadata`、`seq_lens`、`start_pos` 等）。
A→B→A 三步每步都是图重放与 eager 逐 bit 相同、写保护按目标状态自身 slot 计算且 PASS、
输出无非有限值；B 另与在其自身张量上的直接调用逐 bit 相同，第二次 A 与第一次 A 逐 bit 相同。
`compact_slots` 由服务路径在图内重算，不参与只读检查。
修过一个脚本缺陷：遍历 metadata 时对类对象取 `vars()` 会拿到 `__abstractmethods__`
而抛 `AttributeError`，已跳过类/模块/函数并给 `getattr` 加保护。
**边界**：每步都从同一初态开始，因此不覆盖连续多步 decode 轨迹。

## 15. v18（gather 补 early resolve）与 8:2 加权口径

v17 泳道显示核内已无明显空间：`hca_unified_attention` 单块 145 μs、跨度 154.6，
Native 的 `SparseAttnSharedkv` 是 206.1；proj_a 80.4 对 82.8；mHC pre 49.9 对 73.4。
剩下的是阶段边界空档：启动 33 μs，加上 Q_B→Q 反量化 17 μs、Q 反量化→unified attention
13 μs 及其余边界，合计约 113 μs，占整层 18%，而 128K/B16 距 0.8× 差 101 μs。

v18 给关键链上缺 `allow_early_resolve` 的 gather/merge 任务补上
（`hca_gather_kv`、`hca_cmp_work_gather`、`hca_stream_merge_pack`，三条路径共 7 处）；
重任务本来都有。ABBA 净收益：8K +5.91、128K/B16 −3.40、128K/B24 −3.42 μs。
按 128K:8K = 8:2 加权为 −1.55 μs，属边际正收益但在噪声带内，**本节暂不并入 v17，待复测**。

## 16. 下一步

1. 整机 TP1×DP/EP16 的优化版 token 验证，先做 128K/B24（`run_hca_decode.sh` 已支持第 6 个
   参数覆盖 `gpu_memory_utilization`，两侧必须同值），再按用户要求退役 8K/B16。
   **9.2 口径的 Native B24 基线尚不存在**（现有 22.90 GiB / 88 727 us 是 9.0 的），
   整机档期要同时跑 Native 才能给出可比对照。
2. 复测 v18，并在 v17 的 NZ 路径上重测 v14 的 N256（按 8:2 加权它当时是 −4.21 μs）。
3. 关键链上剩余的结构性空间是把 O 投影与 attention 重叠：现在 O 投影严格串在 attention
   之后（128K 下 attention 218→465、O 投影 468→583），而 attention 有 246 μs 的窗口。
   merge/pack 的 `merge_h_tile`（H_TILE=16 头）恰好对应 2 个 O 组
   （`HEADS_PER_GROUP=8`、`PUBLISH_GROUPS=2`），按 h_tile 拆成 4 个任务后
   proj_a 的 8 个组任务可各自只依赖对应的 merge 任务。
4. 固定归约下的连续 decode 状态轨迹、B1～B64 动态 BS 切换、整算子 padding/dummy、
   请求生命周期与 prefix 共享（当前 dummy 证据仅覆盖 compressor 子链）。

## 17. 档位表调整：新增 128K/B24，去除 8K/B16（2026-09-29）

用户定档：**8K/B16 这个档位之后去除。** 验收集由此变为七档：
128K×B4/B8/B16/B24 + 8K×B24/B32/B40。`submit_hca_seven_tier.sh`（2026-09-29 已更名为 `submit_hca_tiers.sh`）的档位串已改为
`131072:4 131072:8 131072:16 131072:24 8192:24 8192:32 8192:40`，队列里两个 8K/B16
的候选任务已撤销。8K/B16 在 v17 下的历史值为 Native 538.1 / PTO 528.7 / 比值 0.983，
只作记录，不再参与验收。

按新七档，v17 的比值均值为 0.963（旧八档是 0.965），最好 128K/B8 的 0.930，
最差 8K/B40 的 1.000。

## 18. ⚠ Native 基线未对齐上线 decode 口径，全部重取（2026-09-29）

用户指出"你的 native 测试需要对齐一下"，并给出事实来源
`vllm-ascend-main/tests/dsv4_perf_accuracy_20260827/runtime/decode/run_dp_template.sh`。
逐项核对后，我的单层 bench 与上线口径的差异如下（左为模板，右为我的旧配置）：

| 项 | 上线模板 | 旧的单层 bench |
| --- | --- | --- |
| `ascend_compilation_config.enable_npugraph_ex` | true | 未配 |
| `ascend_compilation_config.enable_static_kernel` | true | 未配 |
| `enable_cpu_binding` | true | 未配 |
| `multistream_overlap_shared_expert` | true | 未配 |
| `recompute_scheduler_enable` | false | 未配（等效 false） |
| `--max-num-seqs` | 32 | 40 |
| `--max-num-batched-tokens` | 400 | 256 |
| `--enable-prefix-caching` | 开 | 关 |
| `HCCL_OP_EXPANSION_MODE` | AIV | 未设 |
| `HCCL_BUFFSIZE` | 1800 | 未设 |
| `OMP_NUM_THREADS` / `OMP_PROC_BIND` | 10 / false | 未设 |
| `VLLM_BATCH_INVARIANT` | 0 | 未设（默认 0） |
| `VLLM_ASCEND_ENABLE_NZ` | 2 | 2（一致） |

已对齐的改动：`dsv4_hca_single_layer.py` 的 `additional_config` 补齐四项并把
`max_num_batched_tokens` 改为 400、`enable_prefix_caching` 改为 True；
`run_hca_single_layer.sh` 补齐上述五个环境变量。两处都是进程级或共用同一个 config
对象，**Native 与 PTO 必然同配**，不存在单边优待。

有意保留的三处偏离及理由：

1. `max_num_seqs` 取 `max(40, batch)` 而非模板的 32。档位表里有 8K/B32、8K/B40，
   B40 超出模板容量。**这是档位表与上线容量的真实冲突，B40 是否仍应作为验收依据待定。**
2. `enforce_eager=True` 保留。单层 bench 自己捕获 NPUGraph 再用图外 NPU Event 计时，
   为的是把单层隔离出来；模板的 `cudagraph_mode: FULL_DECODE_ONLY` 是整模型口径。
   代价是 `enable_npugraph_ex` / `enable_static_kernel` 在单层里可能空转，只能实测确认。
3. `--async-scheduling`、`--max-model-len 1048576`、`--no-disable-hybrid-kv-cache-manager`、
   `kv_transfer_config` 未配：后三者会改 block_table 宽度与 KV 管理，而单层 fixture 的
   页表宽度是按档位算的，强行换成 1M 会另引入一个变量。

**口径影响：第 13 节与第 17 节的七档表是旧配置下测的 Native 分母，不与新数据混用。**
用户定的处理方式是"新的测试用新的 native 基线就好，之前的不管"，因此不回补旧数据，
新基线从 `seven_aligned_v17/` 开始。

## 19. v19 否定：NZ 下 proj_a 的 N256 没有带宽可换（2026-09-29）

v19 = v17 + proj_a 列块在 ≤96 档也取 N256，即第 11 节 v14 的想法在正确的 NZ 路径上重测。
ABBA 净收益：128K/B16 **+8.31**、128K/B24 **+0.90 μs**，**否定**。
这同时给 v14 那条线索结案：ND 时 N256 能把 256 字节的连续段翻倍、换来 22% 的核时间，
但 NZ 的列切片本身就是连续的，没有带宽可换，只剩"块数 64→32、24 AIC 填充率 89%→67%"
的损失。**proj_a 的列块维度到此关闭。**

## 20. v20 / v18 保留并合成 v21（2026-09-29）

两个候选都只动 `hca_cmp_work_gather`，但分别是块数与调度属性，可以叠：

| 候选 | 改动 | 128K B16 | 128K B24 |
| --- | --- | ---: | ---: |
| v20 | 长档 gather 每块承担的搬运项数按实际项数反推，块数压进一个 AIV 波（48 块） | −3.73 | **−12.26** |
| v18 | `hca_gather_kv`、`hca_cmp_work_gather`、`hca_stream_merge_pack` 补 `allow_early_resolve` | −2.71 | −2.73 |

v20 的依据是实测的波量化浪费：128K/B16 下该任务 72 块、核时间 900 μs
（48 核下限 18.8 μs），跨度却是 51.1 μs，是 1.5 波；同窗口的
`qproj_dequant_rms_nope_rope` 是 48 块、核时间 952 μs（下限 19.8），跨度 73.0。
两者合计只有 38.6 μs 的活却占了 73 μs 窗口，AIV 利用率约 53%。
B24 收益更大与预期一致——搬运项数越多，半波浪费越明显。
v18 首测在 8K/B16 是 +5.91 μs，但该档已退役；128K 两档复测为 −2.71 / −2.73，方向一致。

`source_combo_v21` = v20 叠 v18，CPU 完整编译 PASS，三档 ABBA（128K B16/B24 + 8K B24）
已在新对齐口径下排队。

## 21. 已关闭的方向（避免重复尝试）

- **按 O 组拆 attention 让 proj_a 早开工**：不可行。`hca_unified_attention` 是
  每块处理一个 token、块内一次加载全部 64 个头（`query = pl.load(q_flat, [token * H, 0],
  [H, HEAD_DIM], target_memory=pl.MemorySpace.Mat)`），KV 在一个 token 内被 64 个头共享、
  只流式读一遍。按头组拆会让每份重读整段 KV，128K 下那是主要流量，
  4 倍 KV 换约 100 μs 的重叠不划算。
- **proj_a 的行块维度**（第 11 节 v13）与**列块维度**（第 19 节 v19）都已关闭。
- **HCA 自己的 ring arena 下限探测**：不需要。PTO 在 128K/B24 上此前差 Native 1.63 GiB，
  与第 12 节 wo_a recast 的 1.603 GiB 几乎完全吻合，布局修正后缺口已消失，
  该档位单卡已跑通。用户明确"都跑通了你还探测什么 ring 的下限，没有必要"。
- **kernel 模式约 80 μs 的启动固定开销**：属公共优化，本会话不处理（第 4 节）。

## 22. harness 真正对齐上线口径：enforce_eager 会让平台强关编译开关（2026-09-29）

第 18 节把 `additional_config` 对齐后，我曾据"Native 分母只动 −3～+15 μs"写下
"`enable_static_kernel`／`enable_npugraph_ex` 在单层里基本空转"。用户指出
**单层用 enforce_eager，理论上可以使用这两个开关**。查下去找到确切机制：

`vllm_ascend/platform.py` 在 `compilation_config.cudagraph_mode == CUDAGraphMode.NONE`
时**显式把 `enable_npugraph_ex` 与 `enable_static_kernel` 置 False**，并同时改写
`vllm_config.additional_config["ascend_compilation_config"]` 里的值；而
`enforce_eager=True` 正好让 cudagraph_mode 变成 NONE。所以我写进去的 True 被静默覆盖。
另外两个默认值：`enable_npugraph_ex` 默认 **True**、`enable_static_kernel` 默认 **False**
（`ascend_config.py`），因此"两者都设 True"实际只引入 static_kernel 一个变量。

我原来那句结论的证据也不成立：同卡 Native 自身漂移带是 ±25 μs 量级，而据以下结论的
差异只有 3～15 μs，实验没有分辨力。已改为只说"开关已确认生效，当前实验无法分辨其效果量级"。

改动：`dsv4_hca_single_layer.py` 去掉 `enforce_eager=True`，改为
`compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"}`（上线模板口径；本 bench 仍
自己捕获 NPUGraph 做单层计时）；并新增 `effective_compilation` 报告字段，回显平台
**最终生效**的 `cudagraph_mode` / `enable_npugraph_ex` / `enable_static_kernel` /
`fuse_norm_quant`，以后不再靠看自己传进去的值判断。

探针（128K/B16、v17）回显 `{"cudagraph_mode": "FULL_DECODE_ONLY",
"enable_npugraph_ex": true, "enable_static_kernel": true, "fuse_norm_quant": true}`。
注意 `fuse_norm_quant` 在单层这条路可以开，而整机 offline_pd 因本机缺
`aclnnAddRmsNormBias` 显式关掉了它——单层反而更接近上线口径。

### 新的正式验收基线（`seven_fulldecode_v17/`，static_kernel 七档均确认为开）

| 档位 | Native P50 | 基线 PTO | v17 PTO | v17/Native | 目标 0.8× | 还差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128K B4 | 404.1 | 451.0 | 403.8 | 0.999 | 323.3 | +80.5 |
| 128K B8 | 525.3 | 572.3 | 489.3 | 0.932 | 420.2 | +69.1 |
| 128K B16 | 667.6 | 714.0 | 641.4 | 0.961 | 534.1 | +107.3 |
| 128K B24 | 871.5 | 904.1 | 806.2 | **0.925** | 697.2 | +109.0 |
| 8K B24 | 685.6 | 752.2 | 655.8 | 0.956 | 548.5 | +107.3 |
| 8K B32 | 750.2 | 818.6 | 730.9 | 0.974 | 600.2 | +130.7 |
| 8K B40 | 829.0 | 954.1 | 833.3 | 1.005 | 663.2 | +170.1 |

比值均值 0.965，逐 bit 门禁全 PASS，距 0.80× 仍差 69～170 μs。

## 23. v21 保留：长档 gather 压波 + gather/merge 补 early resolve（2026-09-29）

`source_combo_v21` = v17 + v20（长档 `hca_cmp_work_gather` 每块搬运项数按实际项数反推、
块数压进一个 AIV 波）+ v18（`hca_gather_kv`、`hca_cmp_work_gather`、
`hca_stream_merge_pack` 补 `allow_early_resolve`）。

ABBA 净收益：128K/B16 **−15.94**、128K/B24 **−15.70**、8K/B24 +4.39 μs；
按 128K:8K = 8:2 加权 **−11.78 μs**。保留。

## 24. AICPU dummy 依赖：关键不是 dummy，而是假依赖（2026-09-29）

用户提出：**AICPU 上的 dummy 调度依赖可能比较影响性能，可以去除这种依赖、换成多任务之间
的直接依赖，看 PTO 算子内部的调度性能是否改善。** HCA 路径上共 4 处
`pl.system.task_dummy`：压缩器的 2 路汇聚、O 投影的 8 路汇聚、以及
`decode_hca.py` 与 `qkv_proj_rope.py` 两个空依赖锚点。

| 候选 | 改动 | 128K B16 | 128K B24 | 8K B24 | 8:2 加权 | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v22 | 压缩器返回写 `cmp_cache` 的任务 id，不再用 dummy 汇聚 `state_tid` + `cache_tid` | −4.95 | −10.22 | −3.65 | **−6.80** | **保留** |
| v23 | v22 再去掉 O 投影的 8 路汇聚 dummy | −5.19 | +8.03 | +16.84 | +4.50 | **否定** |

**这两个结果合起来修正了"dummy 一律该去掉"的判断：**

- v22 赢在**消掉一条假依赖**，不在于少一跳。唯一的消费者 `hca_cmp_work_gather`
  只读 `cmp_cache`、并不读压缩器 state，原先的 dummy 把 `hca_state_commit`
  变成了 attention 的前置。128K/B24 收益最大（−10.22），与"B24 的 state 提交更重"一致。
- v23 输在**宽扇入场合 dummy 本身是优化**。`hca_oproj_hc_post` 有 48 个块：带 dummy 时
  每块只跟踪 1 条依赖边，去掉后变成每块 8 条、合计 384 条边要在 AICPU 上核对，
  即 dummy 把 N×M 条边压成了 N+M 条。

结论：**攻击面是过宽依赖，而不是 dummy 的数量。** 宽扇入 + 多块消费者的 dummy 应当保留。

### 写法上的三道限制（改依赖时会撞到）

1. `pl.array` 不能跨 `@pl.jit.inline` 边界返回：codegen 报
   `Internal error: GetCppType called for ArrayType`。
2. `return` 里不能用列表推导：tracer 报 `Unsupported expression type: ListComp`
   （同样的推导式作为 `task_dummy(deps=...)` 的实参却是允许的）。
3. `pl.spmd(deps=...)` 的每一项必须是 TaskId 变量或 TASK_ID 数组元素 `arr[i]`，
   元组下标表达式被拒：`deps= entries must be a TaskId variable, a TASK_ID array
   element (e.g. arr[i]), or None`。可行写法是展开成元组返回、消费侧逐个绑名，
   并对组数加显式守卫防止将来静默少传依赖。

## 25. v24 排队中：压缩器的 `late_dep` 是本轮最大的一条过宽依赖

按第 24 节修正后的判据复查依赖锚点，发现 `decode_hca.py` 把 `qa_tid`（QR 投影）
作为压缩器的 `late_dep` 传入，而压缩器的首个任务 `hca_kv_score_proj` 只读
`normalized`、`cmp_wkv`、`cmp_wgate`，**并不读 qr**。v17 泳道上的代价清晰可见：
Q_A 在 122→135，压缩器在 141→213；若压缩器能在 mHC pre 结束（111）后即起跑，
整条"压缩器 → `cmp_work_gather` → attention"链可提前约 30 μs。

参数名为 `late_dep` 说明这是**故意延后**（CSA 侧也有同名概念，用于错开 cube 争用），
所以它是有意设计还是残留只能实测。v24 = v23 去掉该依赖（改传无依赖锚点；`normalized`
是普通 `create_tensor`，数据依赖由框架自动跟踪，正确性不依赖这个锚点）。
CPU 编译 PASS，三档 ABBA 与一份 128K incore 已排队。

注意 v24 是叠在**已被否定的 v23** 上做的增量对照，其净收益只说明"去掉这条过宽依赖"
本身的效果；若成立，正式版本应当是 v22 + 该项，而不是 v23 + 该项。

## 26. v24 / v25 否定：压缩器的 `late_dep` 是有意延后，且是对的（2026-09-29）

按第 25 节把压缩器的 `late_dep`（原为 QR 投影的 `qa_tid`）改成无依赖锚点，
两种叠法都实测：

| 候选 | 基线 | 128K B16 | 128K B24 | 8K B24 | 8:2 加权 | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v24 | v23（已否） | −15.23 | +12.17 | +14.11 | +1.60 | 否 |
| v25 | v22（保留版） | −6.10 | **+29.02** | **+38.45** | **+16.86** | 否 |

提前起跑确实省下约 30 μs 的等待，但 `hca_kv_score_proj` 是 cube 矩阵乘，提前后与
Q_A/Q_B 抢 cube；B24 与 8K 下这个争用的代价远大于抢到的先手。**参数名 `late_dep`
名副其实，该依赖是有意的错峰设计，保留。**

差值分布（B16 −6 而 B24 +29、8K +38）方向不一致且幅度悬殊，不属于"顾此失彼"，
不构成按档位分支的理由。

**这与第 24 节合起来给出一条判据：过宽依赖要分两类看。**
若被依赖的任务与消费者争用**不同**的执行资源（如 `hca_state_commit` 是 AIV、
消费者 `hca_cmp_work_gather` 也是 AIV 但二者无数据关系），去掉它是纯收益（v22）。
若去掉之后会让两个**同资源**的重任务重叠（压缩器的 cube 与 Q 投影的 cube），
那条"过宽"依赖其实在做错峰，去掉反而更差（v24/v25）。判断前先看两侧各占哪类核。

## 27. 本轮保留版 v22 落地与七档验收（2026-09-29）

确定保留的是 **v21 + v22**，累计相对 v17 约 −18.6 μs（8:2 加权）：

- v21 = v17 + 长档 `hca_cmp_work_gather` 块数压进一个 AIV 波 + `hca_gather_kv`／
  `hca_cmp_work_gather`／`hca_stream_merge_pack` 补 `allow_early_resolve`（−11.78）。
- v22 = v21 + 压缩器直接交出写 `cmp_cache` 的任务 id，消掉 attention 对
  `hca_state_commit` 的假依赖（−6.80）。

已落进生产算子（`deepseek_v4_flash_hca/decode_sparse_attn_hca.py` 与
`decode_compressor_ratio128.py`），生产路径 CPU 完整编译 PASS，
七档验收（`seven_fulldecode_v22/`）已排队。

被否定并已丢弃的候选：v13、v14、v15、v16、v19（proj_a 的行块与列块维度，两维均已关闭）、
v23（O 投影的宽扇入 dummy 应保留）、v24、v25（压缩器 `late_dep` 应保留）。

## 28. ⚠ 计时口径的第三次纠正：ABBA 净收益的噪声底约 ±15～20 μs（2026-09-29）

**本节作废第 20、23、24、26、27 节中所有基于 ±20 μs 以内净收益做出的取舍判断。**

v26 在压波上加了守卫 `cmp_gather_count > CMP_GATHER_AIV_WAVE(48)`。128K/B8 的项数约 64、
B24 约 192，**两者都满足条件，因此 v26 在这两档上执行的代码与 v22 完全相同**。
然而 ABBA 报出的净收益是 B8 **+20.13**、B24 **+11.35** μs。

两个行为完全一致的版本被测出 11～20 μs 的差距，这个数就是本方法的噪声底。旁证：

- 8K/B32 的 v17→v22 净收益，首测 −15.62、独立复测 **+16.99**（摆动 32 μs）。
- 128K/B8 的同一对照，首测 +7.36、复测 +3.02。
- 各轮 Native 漂移量本身在 −26.62～+16.92 之间，与待测效应同量级。

**因此这些结论都不成立**（数值仅作历史记录，不可用于排序）：
v18（−1.55）、v20（−3.73/−12.26）、v21（−11.78）、v22（−6.80）、v23（+4.50）、
v24（+1.60）、v25（+16.86）、v26（B4 −14.19）。它们全部落在噪声带内。

**本会话唯一站得住的性能结论是第 12 节的 v17**，因为它有不依赖端到端计时的独立佐证：
incore 数据显示 proj_a 核时间 2624→1408 μs（−46%）、单块均值 41.0→22.0 μs、
跨度 129.2→75.3 μs，整个 O 投影组由落后 Native 23.5 μs 转为领先。
核内量（核时间、单块均值、块数）比整层 P50 稳定得多，应当作为主判据。

### 已排队的零假设实验

`abba_null_v22_h*/`：baseline 与 candidate 都指向 `source_combo_v22`，按构造净收益必须为 0，
实测离散即噪声底。覆盖 128K B4/B8/B16/B24 与 8K B32 五档。

### 后续判据（在噪声底量化之前不要再叠候选）

1. **以 incore 核内量为主判据**：核时间、单块均值、块数、块数与核数的整除关系。
   只有当一个改动在这些量上有可解释的变化时，才去看端到端 P50。
2. **端到端只用于确认方向**，且效应必须显著大于噪声底；小于噪声底的差异一律记为
   "无法分辨"，不得写成收益或退步。
3. 若确需分辨小效应，先提高精度：加大 `--timing-iters`、每档重复多次独立 ABBA 取中位数、
   或固定同一张卡连续跑完整组对照，而不是靠单次四轮。

### 对已落地内容的处置

v21 + v22 已落进生产（`decode_sparse_attn_hca.py`、`decode_compressor_ratio128.py`）。
它们的改动在机制上是合理的（gather 块数与 AIV 波对齐；消掉 attention 对
`hca_state_commit` 的假依赖，后者是真实的依赖关系修正而非调参），且逐 bit 门禁全 PASS，
因此**不因噪声结论而回退**；但**不得宣称它们带来了量化收益**。
v26 的守卫同理：它在项数不足一个波时退回原分法，属于消除无意义分支，机制正确，
与收益数字无关。

### 28.1 零假设实验结果：噪声底已量化

`abba_null_v22_h*`，两侧都指向 `source_combo_v22`，净收益按构造为 0：

| 档位 | PTO 四轮 P50 | Native 漂移 | 实测净收益 |
| --- | --- | ---: | ---: |
| 128K B4 | 398/402/413/406 | −2.16 | +8.07 |
| 128K B8 | 510/503/504/490 | +3.75 | −0.30 |
| 128K B16 | 643/658/627/638 | +14.60 | −12.37 |
| 128K B24 | 792/781/797/783 | +14.21 | −12.84 |
| 8K B32 | 737/738/730/734 | −2.32 | +0.71 |

样本标准差 **9.05 μs**，绝对值上界 **约 13 μs**（v26 在 B8/B24 上出现过 +20/+11，
说明偶发更差）。因此：**单次 ABBA 只能分辨约 25 μs 以上的效应。**

v17 的 128K/B16 净收益 −41 μs 约为噪声底 3 倍，且有 incore 佐证，结论成立。
v25 被否依据的是 B24 +29、8K +38，同样超出噪声底，否定成立。其余候选均不可分辨。

**加大 `--timing-iters` 解决不了**：主导项是轮次间的热漂移（Native 自身漂 ±14 μs），
不是采样噪声。而两个 PTO 版本必须在不同进程里跑（`@pl.jit` 编译时重读源文件），
无法像 Native/PTO 那样在同一进程内细粒度交替。要把标准差降到 5 μs 需每档重复 4 次
独立 ABBA 取聚合值，代价 4 倍设备时间。

### 28.2 策略调整

缺口是 76～185 μs，而噪声底 13 μs——追 10 μs 量级的调参不是正确打法。改为：

1. 只做单项期望 **≥25 μs** 的结构性改动。
2. 主判据是 incore 核内量（核时间、单块均值、块数与核数的整除关系），
   端到端 P50 仅用于确认方向。
3. 结构性目标按 v17 的 incore（128K/B16，整层 629 μs）排序：

| 目标 | 跨度 μs | 核时间/核数下限 μs | 可回收 |
| --- | ---: | ---: | ---: |
| `qproj_dequant_rms_nope_rope` | 73.0 | 19.8 | ~53 |
| `hca_cmp_work_gather` | 51.1 | 18.8 | ~32 |
| 阶段边界空档合计（不含启动 33） | ~80 | — | — |

首选：把 dequant 融进 `hca_unified_attention`。attention 本就按 token 逐块加载 q
（`query = pl.load(q_flat, [token * H, 0], [H, HEAD_DIM], target_memory=pl.MemorySpace.Mat)`），
改成加载 INT32 的 `q_proj_i32` 并在块内完成反量化 + head RMSNorm + RoPE，
可整段去掉 73 μs 的阶段与两侧约 30 μs 的边界，代价是 952 核时间搬进 attention
（48 块各 +19.8 μs，跨度 154.6→约 175）。期望净 −60～−80 μs，远超噪声底。
风险：算术必须逐 bit 复刻现有 dequant，否则逐 bit 门禁不过。

## 29. 资源占用分析：这一层是依赖链受限，不是算力受限（2026-09-29）

**本节同时推翻我在第 15、28.2 节里"阶段边界空档约 80～113 μs"的说法。**
那些"空档"是按功能组的起止时刻相减得到的，而组与组是重叠的——期间一直有别的任务在跑。

从 v17 的 128K/B16 泳道（`incore_cann92_v17_h131072_b16/after_native`）逐微秒统计
pid 4 的块占用，真实情况是：

- **完全没有任何块在执行的窗口只有 64 μs**：0→31（启动，公共开销）、48→52、56→59、
  81→89、297→309、583→589。除启动外可动的只有 **33 μs**。
- 最大并发 76 块；并发 ≤4 块的时间合计仅 10 μs。

### 真正的瓶颈：核时间远小于跨度

72 个并行通道（24 AIC + 48 AIV），核时间总和 19644 μs，**每通道最多只有 325.3 μs**
（中位 267.1）。按核类型折算：

| 量 | 值 |
| --- | ---: |
| AIC 核时间 ÷ 24 | **383 μs** |
| AIV 核时间 ÷ 48 | 284 μs |
| 资源下限 max(AIC, AIV) | **383 μs** |
| 实测整层跨度 | 625 μs |
| **依赖链造成的额外时间** | **≈ 242 μs** |
| 目标 0.8×Native（128K/B16） | 538 μs |

**结论：目标在资源上可达，余量 155 μs；缺口全在依赖链，不在算力。**
这也解释了为什么第 20～27 节那些调参候选都测不出效果——它们改的是核内耗时或单个
任务的分块，而瓶颈根本不在那里。

### 按 50 μs 分桶的两类核占用

| 区间 μs | AIC | AIV | 主要任务 |
| --- | ---: | ---: | --- |
| 0–50 | **0%** | 10% | 启动、`hca_hc_widen_rms`、`q_rope_prepare` |
| 50–100 | 32% | 18% | `hc_pre_linear`、`hca_warm_kv_weights` |
| 100–150 | 99% | 18% | `qr_proj_matmul`、`hca_kv_score_proj` |
| 150–200 | 85% | 12% | `qproj_matmul`、`hca_raw_cache_write` |
| **200–300** | **2–8%** | 49–54% | **`qproj_dequant_rms_nope_rope`、`hca_cmp_work_gather`** |
| 300–450 | 80–100% | 80–100% | `hca_unified_attention`（两类核都满，理想段） |
| 450–550 | 100% | 26–33% | `proj_a_mm`、`quant` |
| 550–600 | 54% | 24% | `_proj_b_mm_nz_kernel_`、`hca_oproj_hc_post` |
| 600–625 | **0%** | 85% | `hca_oproj_hc_post` |

### 据此确定的优化方向（按期望收益排序）

1. **填 200–300 μs 的 AIC 空窗（期望 ~100 μs）。** 那 100 μs 里 AIC 只有 2–8% 占用，
   而 AIC 是瓶颈资源。该段在跑 dequant 与 cmp_gather（都是 AIV 活），
   attention 的 AIC 部分要等 300 μs 才开始，因为它等 dequant 产出的 q。
   **把 dequant 按 token 融进 `hca_unified_attention`**，attention 的 AIC 就能在约
   200 μs 起对已就绪 token 开工。期望收益是第 28.2 节估的 60～80 μs 的上修版本，
   且这次有资源占用数据支撑，也远超 13 μs 噪声底。
   已知风险：dequant 的 RMSNorm 用 `pl.row_sum` 在 [8 token × N head] 分块上做，
   融合后分块高度变化可能改变行内归约顺序、破坏逐 bit 门禁。绕法是让 attention 块
   改取**连续** token 段并保持 T=8 分块（现有代码已有处理不满 8 行的尾部分支）。
2. **让 `hca_oproj_hc_post`（600–625，AIV 85%、AIC 0%）与 `proj_a_mm`（450–550，
   AIC 100%、AIV 26%）重叠**：两者核类型互补，但 mHC post 依赖 proj_b，需要按组流水。
3. 0–100 μs 段两类核占用都极低（AIC 0→32%、AIV 10→18%），是第二块可填的窗口；
   但该段含 31 μs 启动（公共开销），实际可动的空间较小。

**判据**：以上都用 incore 的两类核占用与整层跨度判断，不再依赖单次 ABBA 的 P50 差值。

## 30. 填 AIC 空窗的四条路径与第 A 条的阻塞点（2026-09-29）

按第 29 节，200–300 μs 那 100 μs 里 AIC 只有 2–8% 占用，而唯一能填它的是 attention，
attention 又同时等 q（dequant 产出）与 `cmp_work_kv`（cmp_gather 产出）。四条路径：

| 路径 | 机制 | 期望 | 风险／状态 |
| --- | --- | ---: | --- |
| A | 串联 dequant 与 `hca_cmp_work_gather`，两者各吃满 48 个 AIV 通道，dequant 跨度从 73 回到 19.8 μs 下限 | −50 μs | 数值天然不变；**被 codegen 阻塞，见下** |
| B | dequant 按 token 融进 `hca_unified_attention`，AIC 约 200 μs 起开工 | −100 μs | `row_sum` 分块高度变化可能破坏逐 bit 门禁 |
| C | 用 FIXPIPE 把反量化缩放并入 Q_B 的 cube 收尾 | −10～15 μs | RMSNorm 是整行归约，做不进 FIXPIPE；只能省掉 INT32 的一次回写读回 |
| D | 加宽压缩器让 `cmp_work_gather` 更早开始 | 有限 | 压缩器块数 = `pl.min(requests, WORKERS)` = 请求数，B16 下就是 16，受设计所限 |

补一个量化结论：**dequant 的 952 核时间是带宽下限**（约 18 MiB 搬运 ÷ 19.8 μs ≈ 0.9 TB/s），
不是可压缩的计算量。所以 A 最多把跨度从 73 压回 19.8，不可能更少。

### A 的阻塞点：task id 不能穿出 `pl.range` 追踪循环

试做探针 `source_probe_dqtid`：把 `q_proj_q_dequant` 改成 `with pl.spmd(...) as dq_tid`
形式并返回 tid，再经 `q_proj_q` → `q_proj_rope` → `qkv_proj_rope` → `decode_hca` 串给
长档的 `hca_cmp_work_gather`。Python 语法与各层签名都通，但编排编译失败：

```
_decode_hca_tp1_layer.cpp:413:50: error: 'dq_tid_inline675_in…' was not declared in this scope
```

原因是 dequant 建在 `q_proj_q` 的 `for tile_base in pl.range(0, t_dim, PREFILL_DENSE_TILE)`
里，生成的 C++ 把该变量放在循环体作用域，循环外不可见。**这是第四道需要记住的写法限制**
（前三道见第 24 节）：`pl.range` 追踪循环内创建的 TASK_ID 无法在循环外使用。

绕开需要把 dequant 提到 tile 循环之外，连带把 `q_proj_i32` 从 per-tile scope 提出来
（decode 下容量约 30 MiB，可接受），但这会改动 CSA 共用的 `q_proj_q` 结构，
必须同时保证 prefill 的多 tile 行为不变。**下一步就做这个重构，或转做路径 B。**

## 31. 根因定位：整层跨度的主体是"每块派发摊开"，不是计算也不是闲置算力（2026-09-29）

### v27 的否定结果与它给出的诊断

v27 = v22 + 长档 `hca_cmp_work_gather` 追加依赖 `q_ready_dep`（dequant 的任务 id），
目的是把两个 AIV 任务串起来、各自吃满 48 通道。实现上打通了第 30 节的阻塞点，
办法是沿用本文件 `q_proj_qr` 的既有惯用法：**用一元 `pl.array` 承载 TASK_ID 带出
`pl.range` 追踪循环**（`return dq_tids[0]`）。这同时澄清了第 24 节的限制 1——
不能返回 `pl.array` 本身，但可以返回它的**元素**。顺带修掉一处无用返回：
`q_proj_q_dequant` 原本 `return q`，而 `q` 是 in/out 参数、返回值被调用点丢弃。

实测（incore 为主判据）：dequant 跨度 73.0 → 39.1 μs，`cmp_work_gather`
72 块/51.1 → 48 块/31.9。但 **200–300 μs 的 AIC 空窗没有被填**
（200–250 从 8% 变 0%，250–300 仍 0%），整层 625 → 633 μs。

原因：串联只是把并行改成接续。v17 下两者并行、合计占 73 μs；v27 下 dequant 216–255、
gather 258–290，合计 71 μs——**一样**。attention 仍在 ~300 才开始。

### 真正的限制：块派发摊开

对 v17 的 128K/B16 泳道按任务统计"最早块起跑到最晚块起跑"的摊开量：

| 任务 | 块数 | 起跑摊开 μs | 单块最大 μs | 跨度 μs | 摊开/块 μs | 摊开占跨度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `proj_a_mm` | 64 | 65.5 | 34.6 | 98.7 | 1.02 | 66% |
| `_proj_b_mm_nz_kernel_` | 64 | 63.0 | 21.2 | 75.5 | 0.98 | 83% |
| `qproj_dequant_rms_nope_rope` | 48 | 53.8 | 22.6 | 73.6 | 1.12 | 73% |
| `hca_cmp_work_gather` | 72 | 36.3 | 17.0 | 51.7 | 0.50 | 70% |
| `quant` | 24 | 27.7 | 10.6 | 89.5 | 1.16 | 31% |
| `hca_kv_score_proj` | 16 | 16.1 | 14.3 | 26.8 | 1.01 | — |
| `qproj_matmul` | 24 | 9.8 | 48.2 | 50.6 | 0.41 | — |

**全层 664 块，各任务起跑摊开合计 309 μs，平均 0.46 μs/块**（大任务约 1.0 μs/块）。

这一个事实解释了本会话的全部现象：

- 第 20～27 节那些改分块、改缓存策略、改依赖顺序的候选都测不出效果——它们不改变
  块数，就不改变摊开总量。
- v27 的串联必然白做——前后排列不改变块总数。
- 第 29 节算出的 242 μs"依赖链余量"填不上——它本身就是这 309 μs 摊开在关键链上的投影，
  而不是可以靠重排回收的闲置。
- 第 28 节那些 ±13 μs 的噪声之所以盖过一切候选效果，是因为候选的真实效应本就接近零。

### 经验模型与算子侧的上限

`span ≈ 块数 × 每块派发 + 单块耗时`。以 `proj_a_mm`（核时间 1421 μs、24 个 AIC、
每块派发约 1.0 μs）代入：

| 块数 | 预测跨度 μs |
| ---: | ---: |
| 24 | 24 + 59 = 83 |
| 48 | 48 + 30 = 78 |
| 64（当前） | 64 + 22 = 86（实测 98.7） |

即算子侧调块数最多再省 10～20 μs，**多数任务的块数已接近最优**。纯计算下限
（核时间 ÷ 核数）是 AIC 317 μs、AIV 251 μs，而整层 625 μs。

### 结论与下一步

**剩余缺口的主体在 PyPTO 运行时的每块派发吞吐上，不在 HCA 算子内。**
664 块 × 0.46 μs = 309 μs；把它减半约得 150 μs，正好是七档距 0.80×Native 的缺口量级
（69～170 μs）。

按本会话确立的范围，这一项**在范围内**：用户明确"本仓的 `pypto/` 与 `simpler/` 代码
也都在优化范围内"，只把"每次调用约 80 μs 的启动固定开销"划为公共优化（第 1、4 节）。
**每块 0.46 μs 的派发开销与那个每次调用的固定开销是两件不同的事。**

下一步：读 `simpler/` 的编排器（`src/a2a3/runtime/tensormap_and_ringbuffer/runtime/orchestrator.h`
等），定位单块派发这 0.46～1.1 μs 花在哪里。

## 32. ⚠ 纠正第 31 节：每块派发不是常数，摊开来自"就绪时空闲核不足"（2026-09-29）

**第 31 节里"全层 664 块、纯派发 221 μs、平均 0.33 μs/块"的表述是错的**，用错了模型。
扫 `Q_DEQUANT_WORKERS` 得到的实测直接否定了"每块派发是常数"：

| dequant 块数 | 核时间 μs | 单块均值 | 起跑摊开 | 每块派发 | 跨度 | 线性模型预测 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 48（原值） | 952 | 19.8 | 53.8 | **1.12** | 73.6 | 73.6 |
| 32 | 1050 | 32.8 | **10.7** | **0.33** | **43.8** | 68.7 |
| 24 | 857 | 35.7 | 14.3 | 0.60 | 49.7 | 62.6 |

每块派发从 1.12 掉到 0.33，说明它随块数变化；线性模型在 N=32 处预测 68.7 而实测 43.8，模型不成立。

### 正确的机制

摊开发生在**任务需要 N 个核、而它就绪时空闲核不足 N 个**的时候。dequant 原本要 48 个
AIV，而同窗口的 `hca_cmp_work_gather` 正占着一部分，于是它的块只能随核陆续空出而逐个起跑。
降到 32 块后能塞进当时空着的核里，摊开塌掉。

**"索要满核"本身无害**，这一点有对照：`hca_unified_attention_aic`（24 块 = AIC 总数）与
`hca_oproj_hc_post`（48 块 = AIV 总数）的摊开只有 2.8 与 2.7 μs，因为它们就绪时对应阵列
是空的。所以第 31 节"减块数"的笼统建议也不对——要看的是**并发任务的块数之和与核数的关系**。

### v28：两个并发 AIV 任务各留余量

`source_combo_v28` = v22 + `Q_DEQUANT_WORKERS` 48→32 + `CMP_GATHER_AIV_WAVE` 48→32。

| 量 | v22 | v28 |
| --- | ---: | ---: |
| dequant 摊开 | 53.8 | **19.6** |
| dequant 止 | 296.9 | 282.4 |
| `hca_cmp_work_gather` 块数/摊开 | 72 / 36.3 | 29 / 32.9 |
| **attention 起跑** | 309.6 | **292.8** |
| mHC post 结束 | 624.2 | 614.3 |
| 整层 DFX | 629.0 | 620.2 |
| 总块数 | 664 | 605 |

端到端 ABBA：128K/B16 −2.49、128K/B24 −3.49、8K/B24 −10.88，8:2 加权 **−4.57 μs**，
**在 ±13 μs 噪声底内、不可分辨**。整层泳道跨度本身在这几次运行间也有 ±8 μs 离散
（v17 629.0、dqw32 635.4、dqw24 631.9、v28 620.2），因此整层那一项同样不单独定论。

**处置：保留 v28，只声称机制成立、不声称量化收益。** 依据是 incore 的结构量有独立证据
（摊开 53.8→19.6、attention 起跑提前 16.8 μs 是可重复观测的结构量，不是 P50 差值），
且三档端到端同向为负、无一档变差。

### v29：让并发块数之和正好等于核数（排队中）

按上面的机制，dequant 与 gather 并发争 48 个 AIV，v28 下两者是 32+29=61 仍然超。
`source_combo_v29` 取各 24 块、合计 48 正好铺满。CPU 编译 PASS，incore 已排队。
这是该机制的直接预测：若 dequant 与 gather 的摊开同时降到个位数、attention 再提前，
机制成立；若不降，则"空闲核不足"也不是全部成因，需再找。

### 量级提醒

即使机制成立，v28 拿到的是 9～17 μs 量级，而七档距 0.80×Native 的缺口是 69～185 μs。
**这条线索解释的量级不足以达标**，不应据此暗示目标可达。

## 33. v29 否定，并纠正第 28.2 节的判据：泳道跨度不是端到端时间的可靠代理（2026-09-29）

v29（dequant 24 块 + gather 24 块，合计 48 = AIV 核数）的 incore 完全兑现了第 32 节的预测：

| 版本 | 并发块数之和 | 摊开之和 | attention 起 | 整层 DFX |
| --- | ---: | ---: | ---: | ---: |
| v22 | 120（48+72） | 90.1 | 309.6 | 629.0 |
| v28 | 61（32+29） | 52.6 | 292.8 | 620.2 |
| v29 | **48（24+24）** | **5.3** | **289.3** | **612.5** |

摊开之和塌了 17 倍，attention 提前 20.3 μs，泳道整层 −16.5 μs。**但端到端 ABBA 是**：
128K/B16 **+23.75**、128K/B24 **+18.79**、8K/B24 +6.89，8:2 加权 **+18.40 μs**，
三档全部变差且超出 ±13 μs 噪声底。

### 结论与判据纠正

**第 28.2 节"以 incore 核内量为主判据"的说法要收窄。** 正确的区分是：

- **核时间（core time、单块均值）稳定，可以做判据。** 例如 v17 的 proj_a 核时间
  2624→1408 μs，那是真实且可重复的核内量，端到端也兑现了 −41 μs。
- **泳道里的跨度（span）与起跑摊开（spread）不是端到端时间的可靠代理。**
  泳道是连续回放（权重热、并发形态固定），正式计时是交替跑 Native/PTO，两种口径下同一份
  代码的任务重叠关系不同。摊开在正式口径下未必是约束，而减少块数带来的"每块耗时翻倍"
  却是实打实的：dequant 从 48 块降到 24 块，单块均值 19.8→35.7 μs。

**处置：v29 否定并丢弃；v28 降级为"无可测效果"**（8:2 加权 −4.57，落在噪声底内）。
块数配平这条线到此结束。第 32 节描述的"摊开来自就绪时空闲核不足"作为**泳道现象的解释**
仍然成立且有三点单调证据，但**它不构成端到端优化手段**，不要再据此改块数。

### 本会话在同一现象上的三次表述修正（留作教训）

| 次 | 表述 | 被什么否定 |
| ---: | --- | --- |
| 1 | 每块派发 0.33 μs 是常数，全层 221 μs 派发开销 | 扫块数发现它随块数从 1.12 变到 0.33；线性模型在 N=32 处预测 68.7 而实测 43.8 |
| 2 | 索要满核有害 | `attention_aic`（24=满）与 `oproj_hc_post`（48=满）摊开仅 2.8／2.7 μs——就绪时阵列空闲 |
| 3 | 摊开来自就绪时空闲核不足；配平块数可提速 | 配平后泳道全面改善而端到端反而差 18.4 μs——泳道跨度不是端到端的代理 |

共同点：**我反复用一个口径下的观测量去预测另一个口径下的结果。** 今后对"跨度/摊开/空窗"
一类调度观测量，只能用于解释现象与定位方向，任何收益结论必须回到正式计时口径，
并且效应要显著超过 ±13 μs 噪声底。

## 34. AICPU 的 dummy 调度依赖：实测只有 3～10 μs，不是缺口来源（2026-09-29）

用户提出假设：「aicpu 上的 dummy 调度依赖，可能是比较影响性能的，可以去除这种依赖，
换成多任务之间的直接依赖」。这一节用泳道原始记录把它量掉，结论是**假设不成立**，
并顺带给出真正的量级分布。

### 34.1 dummy 在运行时里的确切语义

`simpler/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler.h:524`：

> Dependency-only tasks (active_mask is empty, shape == DUMMY). Drained by
> the dispatch loop and completed inline — never goes to AICore.

也就是 `active_mask` 为空的纯依赖任务，进 `dummy_ready_queue`，由派发循环 Phase 3
（`scheduler_dispatch.cpp:1204`，`DUMMY_DRAIN_BATCH = 8`）成批弹出并就地退休，
不占 AIC/AIV。`push_ready_routed()`（同文件 `scheduler.h:544`）还会把
**谓词不通过的任务**一并丢进这个队列，所以它同时是"被跳过的任务"的出口。

### 34.2 实测：每层 2～4 个 dummy，合计 2.8～7.0 μs

运行时带 `chip_swimlane_aicpu_record_dummy_task` 埋点，泳道里有任务级记录。
取 `results/hca_optimization_20260928/*/after_native/dfx/`（24 次运行，eager 采集）：

| 项 | 实测 |
| --- | --- |
| 每层 dummy 任务数 | **2～4 个**（24 次运行全部落在此区间） |
| dummy 退休阶段合计 | **2.8～7.0 μs** |
| 占层跨度 | **0.6%** |

以 `incore_v29_h131072_b16` 为例，3 个 dummy 全在调度流 0，退休合计 3.40 μs。

### 34.3 连"它插入的等待"一起算，也还在噪声底以下

退休耗时不等于它给依赖链加的延迟——消费者要等派发循环走到 Phase 3。
用 `chip_swimlane_records.json` 的 `dummy_task` 记录（`clock_freq_hz = 50 MHz`，
1 cycle = 0.02 μs）量"前一阶段结束 → dummy 退休"：

| loop_iter | 前一阶段 | 等待 | dummy 阶段 span |
| --- | --- | --- | --- |
| 22 | （首条，无法测） | — | 2.92 μs |
| 27 | release | 1.64 μs | 0.16 μs |
| 230 | complete | 4.80 μs | 0.32 μs |

调度循环迭代周期 2.61 / 3.48 / 4.07 μs（三条流），与上面的等待同量级——
dummy 基本是在下一两次迭代内就被排空的。

**退休 3.40 μs + 等待约 6.4 μs < 10 μs，而噪声底是 ±13 μs（第 28 节）、缺口是 76～133 μs。**
即使把 dummy 彻底去掉、且全部等待都在关键路径上，也测不出这个改善。

### 34.4 而且"换成直接依赖"方向是反的

dummy 做的是扇入汇聚：N 个生产者 → 1 个 dummy → M 个消费者，共 **N+M** 条依赖边；
改成两两直连是 **N×M** 条。依赖边是在 `complete` 阶段被遍历的
（`on_task_complete(...).fanout_edges`），而 `complete` 是调度器的最大开销项
（见 34.5）。所以去 dummy 会**增加**调度器负担。

这与第 24 节 v23 的实测完全一致：去掉 O 投影的宽扇入 dummy 后，48 块消费者
从每块跟踪 1 条依赖边变成 8 条，端到端更差。v23 是在算子侧做同一件事的实验，
本节是从运行时侧解释它为什么必然更差。

### 34.5 顺带量出的真实分布（`incore_v29_h131072_b16`，层跨度 591.0 μs）

| 调度阶段 | 次数 | 合计 | 占层跨度 | 处理任务数 |
| --- | --- | --- | --- | --- |
| complete | 128 | 448.5 μs | 75.9% | 592 |
| dispatch | 52 | 229.9 μs | 38.9% | 446 |
| early_dispatch | 30 | 86.2 μs | 14.6% | 146 |
| release | 28 | 14.6 μs | 2.5% | 50 |
| resolve | 5 | 11.4 μs | 1.9% | 14 |
| **dummy** | **3** | **3.4 μs** | **0.6%** | **3** |

（三条调度线程并行，合计 134% 超过 100% 是正常的。）

三条主调度线程的并发忙碌分布：0 条忙 35.6%、1 条 18.7%、2 条 22.6%、**3 条全忙 22.8%**。

⚠ **不要把 `complete` 的 448.5 μs 读成"每次完成的固定开销"**：它是
`check_running_cores_for_completion`，主体是**轮询尚未结束的核**，即调度器在等。
把它压小本身不产生收益。这正是第 33 节那个错误的同一类陷阱——
泳道上的忙碌时间不等于关键路径。要判断调度是否真的挡住了执行，
应该量"依赖满足 → 派发到核"的延迟，而不是调度器的忙时。

### 34.6 调度线程数已经拿满，没有这条路

`scheduler_cold_path.cpp:711`：`active_sched_threads_ = aicpu_thread_num_ - 1`
（一条给 orchestrator）。而 `launch_aicpu_num` 的合法范围是
`0 (auto) 或 [2, 4]`（`python/simpler/task_interface.py:125`）。
所以 4 个 AICPU → 3 条调度线程，**当前已是上限**，加线程不可行。

## 35. 核内时间与时间轴全图：最大损失是 AIC 的 102.7 μs 连续空窗（2026-09-29）

第 34 节排除 dummy 后，用同一批泳道把"这 593 μs 到底花在哪"完整量了一遍。
数据源 `results/hca_optimization_20260928/incore_cann92_v17_h131072_b16/after_native/dfx`
（v17，eager 采集，pid 4 Worker View），并用 `incore_v28_h131072_b16` 做一致性对照，
两者结论一致。

### 35.1 核内时间按任务名的分布（128K/B16）

| | AIC（24 核） | AIV（48 核） |
| --- | --- | --- |
| 核时合计 | 8940.8 μs | 13224.4 μs |
| ÷核数 = 完美打包下限 | **372.5 μs** | 275.5 μs |
| 平均占核 | 15.1/24（63%） | 22.3/48（46%） |

单项占比（AIC）：`hca_unified_attention_aic` 38.7%（24 块，均 144.0 μs）、
`qproj_matmul` 11.3%、`qr_proj_matmul` 10.8%、`proj_a_mm_0` 合计约 20%（8 个 task id × 8 块）。

单项占比（AIV）：`hca_unified_attention_aiv` 53.9%（48 块，均 148.5 μs）、
`hca_oproj_hc_post_0` 11.1%、`qproj_dequant_rms_nope_rope` 7.4%、`hca_cmp_work_gather_0` 7.1%。

**attention 一个任务就占 AIC 的 39% / AIV 的 54%，且 24/24 AIC + 48/48 AIV 全占满、
占核率 96%**，所以它的单块时间（≈148 μs）几乎等于它在层跨度里的贡献，已无打包空间。

### 35.2 attention 已按档位分支，不必再造分支

- 128K 走 `hca_unified_attention_*`：AIC 均 144.0 μs / AIV 均 148.5 μs。
- 8K 走 `hca_short_attention_pack_*`：AIC 均 65.0 μs / AIV 均 69.1 μs。

8K/B16 层跨度 464.9 μs，AIC 下限 288.6 μs、AIV 下限 176.0 μs。
**非 attention 部分两档几乎相同**（AIC 下限 224～228 μs），差异全在 attention。

### 35.3 时间轴：三段结构与各段的空闲

| 段 | 窗口 | 实际 | 该段核时下限 | 空隙 |
| --- | --- | --- | --- | --- |
| 头（widen_rms → dequant） | 0 → 265.8 | 266 μs | AIC 109 / AIV 92 | **约 157 μs** |
| attention | 278.5 → 433.5 | 155 μs | 148 μs（96% 占核） | 约 7 μs |
| 尾（proj_a → quant → proj_b → oproj_post） | 419 → 593 | 174 μs | AIC 120 / AIV 51 | 约 54 μs |

### 35.4 ★ 最大单项损失：AIC 连续空闲 102.7 μs

| 部件 | 完全空闲合计 | 最长单个空窗 |
| --- | --- | --- |
| **AIC（24 核）** | 116.3 μs（19.6%） | **175.8 → 278.5，102.7 μs** |
| AIV（48 核） | 107.3 μs（18.1%） | 533.3 → 558.3，25.0 μs |

AIC 的 116.3 μs 空闲里有 **102.7 μs 是一个连续窗口**，占整层 17.3%，比七档验收
还差的 76～133 μs 缺口还大。它的成因在时间轴上一目了然：

```
  qproj_matmul(AIC)          125.2 ──────── 175.8
  [AIC 全空 102.7 μs]                      175.8 ══════════════════ 278.5
  raw_cache_write(AIV)                 152.3 ── 164.0
  norm_rope_write(AIV)                       164.8 ─ 171.4
  state_commit(AIV)                            173.4 ─ 181.4
  inverse_rope_sign(AIV)                         178.0 ─ 185.2
  gather_kv / raw_valid(AIV)                       186.7 ── 202.4
  cmp_work_gather(AIV, 72 块, 占核 39%)               192.1 ────────── 243.8
  qproj_dequant_rms_nope_rope(AIV, 48 块, 占核 36%)   192.2 ──────────── 265.8
  hca_unified_attention_aic/aiv                                    278.5 ──→
```

也就是：这 103 μs 里 AIV 在跑 KV 写回 → gather → dequant 这条链（各任务占核率只有
36%～58%），而 **`hca_unified_attention_aic` 在等 `qproj_dequant_*` 的全部 48 块跑完**，
是一道整体栅栏（265.8 结束，278.5 才起，中间还有 12.7 μs 派发延迟）。

### 35.5 由此确定的方向

**把 gather/dequant → attention 的整体栅栏改成分块流水**：让 attention 的 cube 部分
按压缩块分组，各组只依赖自己那份 gather+dequant，而不是等全部完成。
上界就是那 102.7 μs 空窗。

注意这与第 34 节被否的"去 dummy 改直连"不是一回事：那个是把 N+M 条依赖边变成 N×M 条、
粒度不变；这里是**把一道全局栅栏拆成若干条按块对应的边**，依赖边总数不增加，
减少的是等待。

动手前按既有约束先做两件事：查 pypto-lib 有没有现成的分块流水实现
（`decode_csa.py` 是参考最佳实现），以及查 CSA 会话在
`DSV4_FLASH_CSA_VALIDATION_LOG.md` 里是否已经记过同类优化。

## 36. 缺口的绝对口径：设备跨度要压到 470 μs，其中 68 μs 是不归本项目的 launch 开销（2026-09-29）

第 35 节给出了泳道内部的分布，但要判断"还差多少"必须回到端到端。本节把两者对齐。

### 36.1 七档绝对 P50（v22，`seven_fulldecode_v22/`，图外 NPU Event，50 次）

| 档位 | Native P50 | PTO P50 | 比值 | 目标 0.80×N | 还需降 |
| --- | --- | --- | --- | --- | --- |
| 128K/B4 | 405.5 | 409.8 | 1.011 | 324.4 | 85.4 |
| 128K/B8 | 529.2 | 499.7 | 0.944 | 423.4 | 76.3 |
| 128K/B16 | 672.8 | 643.5 | 0.957 | 538.2 | **105.3** |
| 128K/B24 | 868.2 | 799.8 | 0.921 | 694.5 | 105.3 |
| 8K/B24 | 678.5 | 654.0 | 0.964 | 542.8 | 111.2 |
| 8K/B32 | 744.3 | 728.1 | 0.978 | 595.4 | 132.7 |
| （8K/B40，已退役） | 818.7 | 840.1 | 1.026 | 655.0 | 185.1 |

同目录 `baseline`（v17 之前的保留版）比值 1.05～1.16，可见 v17 起的改动确实把 PTO
从慢于 Native 拉到快于 Native，但离 0.80 还差一截。

### 36.2 ★ 设备跨度与端到端之间有约 68 μs，属于 launch 开销

128K/B16：PTO P50 **643.5 μs**，而同配置泳道的设备侧跨度只有 **约 575 μs**（v22 量级，
见第 35 节 v28/v29 的 575.9 / 574.8）。差额 **约 68 μs** 与已知的每次调用 kernel-mode
launch 下限同量级，用户已明确这属于公共优化、本项目不动。

由此得到真正的工程口径（128K/B16）：

| 量 | 值 |
| --- | --- |
| 目标端到端 | 538.2 μs |
| 减去 launch 开销 | −68 μs |
| **允许的设备跨度** | **约 470 μs** |
| 当前设备跨度 | 约 575 μs |
| 核时下限（attention 148 + 其余 AIC 228） | 约 376 μs |
| 现有打包空隙 | 575 − 376 = **约 199 μs** |
| **必须吃掉的空隙比例** | 105 / 199 = **约 53%** |

结论：**算术上可行，但要把现有打包空隙吃掉一半以上**，不是靠单点微调能到的。

### 36.3 ⚠ 泳道改善 ≠ 端到端改善：v28/v29 的教训必须前置

| 版本 | dequant 块数 | dequant 窗口 | AIC 最长空窗 | attention 起点 | 设备跨度 |
| --- | --- | --- | --- | --- | --- |
| v17 | 48 | 192.2 → 265.8 | **102.7 μs** | 278.5 | 593.0 |
| v28 | 32 | 194.1 → 244.0 | 74.5 μs | 254.4 | 575.9 |
| v29 | 24 | 198.7 → 243.2 | 66.9 μs | 256.7 | 574.8 |

v28/v29 把空窗从 102.7 压到 66.9 μs、跨度压了 18 μs，**但 v29 端到端反而差 18.4 μs**
（第 33 节），v28 的净收益也落在 ±13 μs 噪声底内。所以第 35 节那些空窗数字只能用于
**定位**，任何候选仍必须用 ABBA 端到端判定。本节把这条前置，免得再被泳道误导一次。

### 36.4 attention 起步晚的真实原因是资源，不是依赖

`_long_sparse_attn_hca_tp1` 里 attention 的依赖是
`deps=[raw_gather_tid, raw_valid_tid, cmp_gather_tid, rope_cs_tid]`，
**不含 `qproj_dequant`**。这四个依赖在 243.8 μs 就全部满足，可 attention 到 278.5 才起。

原因是 `hca_unified_attention` 是要同时占 **24 AIC + 48 AIV** 的 MIX 任务，
必须等 AIV 从 gather/dequant 里腾空（dequant 到 265.8 结束）。
这正是 CSA 记过的那条告诫："单独看空闲 AIC 也不能判断 MIX 所需 AIV 是否可用"。

已按该告诫实测空窗内的 AIV 余量（175.8 → 278.5，102.7 μs）：

| 量 | 值 |
| --- | --- |
| 窗口内 AIV 平均占用 | 21.9 / 48 核 |
| **平均空闲 AIV** | **26.1 核** |
| 空闲 AIV 容量 | 2682 核·μs（= attention AIV 需求的 38%） |
| 空闲 AIC 容量 | 2465 核·μs（= attention AIC 需求的 71%） |

即余量确实存在，瓶颈是这段窗口里 AIV 被一条 6 级串行链占着且占核率只有 36%～58%。

### 36.5 v30 探针：gather 不等压缩器能省多少（上界测量）

`hca_cmp_work_gather` 依赖 `cmp_cache_ready_dep`（v22 已收窄为压缩器写 cmp_cache 的
任务 id）。但**压缩器每请求每步只写 1 行**
（`decode_compressor_ratio128.py` 末尾 `cache_flat[cache_row:cache_row + 1, :]`），
而 gather 要搬的是整个压缩历史（128K 下 `cmp_work_count = 8` 个 K128 tile、约 1024 行）。
也就是 **7/8 的搬运量与本步压缩器输出无关**，却被那一行的写入挡住。

v30 探针（`results/hca_optimization_20260929/source_v30/`，**只改长档那一处** `deps=[]`）
用来量收益上界。它数值上不严格，仅取证用，不作为交付（见第 0.2 节口径）。
单层 bench 的 metadata 是定值重放、压缩器每次写同一行，因此精度检查有可能照样通过，
但那不构成正确性证明。

正式做法（若探针显示收益够大）是按 tile 拆成两个任务，判据**只依赖一个可证明的事实**：

- 压缩器本步写入的行索引 k 满足 `k ∈ {gather_rows−1, gather_rows}`
  （`k = (closing+1)//RATIO`，`closing+1 ≤ length`，`gather_rows = length//RATIO`；
  又因 `S = 6 ≪ RATIO = 128`，每步每请求最多跨一个压缩边界）。
- 于是 tile `w` 覆盖行 `[w*128, (w+1)*128)`，**当 `(w+1)*CMP_ATTN_K_TILE < gather_rows`
  时该 tile 的最大行号是 `gather_rows−2`，与 k 必然不相交** → 归入"历史任务"，`deps=[]`。
- 其余（末尾那个不满的 tile）归入"新行任务"，`deps=[cmp_cache_ready_dep]`。

两个任务对 item 的划分是**互斥且穷尽**的，搬运内容、顺序、掩码都不变，
因此是数值中性的改动（按既有约定也应同步到精度版）。

8K 走 `_short_sparse_attn_hca_tp1`，那里 `cmp_work_count = 1`，只有一个 tile、必然含新行，
所以该改动在 8K **不生效也不变差**，符合 128K:8K = 8:2 的取舍口径。
8K/B24 这一档在本轮作为对照档提交。

## 37. v30 否定，并用官方工具拿到权威关键路径（2026-09-29）

### 37.1 v30 否定：栅栏不是问题

| 档位 | PTO 基线 | PTO 候选 | Native 漂移 | 净收益 |
| --- | --- | --- | --- | --- |
| 128K/B16 | 641.03 | 630.80 | +4.15 | **−14.39 μs** |
| 128K/B24 | 797.41 | 798.57 | +9.36 | **−8.20 μs** |
| 8K/B24 | 653.67 | 657.54 | −5.68 | **+9.54 μs** |

按 128K:8K = 8:2 加权后是 **−7.1 μs，落在 ±13 μs 噪声底以内**。

**8K/B24 那一档是一次意外有用的对照**：v30 只改了长档（`_long_sparse_attn_hca_tp1`），
8K 走 `_short_...`，两侧执行的代码**完全相同**，可它测出 **+9.54 μs**。
这独立印证了单档单轮测量的噪声量级就是 ±10 μs 左右，与第 28 节的 ±13 μs 一致。
以后凡是"只在一档上看到十几 μs 收益"的结论，都要当作噪声处理。

因此按 35.5 节设计的正式 v31（历史 tile / 末尾 tile 拆分）**不再实施**：
它的上界就是 v30，而 v30 不可测。第 0.2 节补记。

### 37.2 用 `simpler_setup.tools.critical_path` 拿权威关键路径

本仓 simpler 自带工具，能从 level-4 泳道重建官方关键路径，比我手推时间轴可靠：

```bash
source env-dsv4-0251rc1.sh
cd simpler && python3 -m simpler_setup.tools.critical_path <结果目录>/after_native --stdout
```

已有泳道的 `chip_swimlane_level` 就是 4，**不需要重采**。报告写在
`<...>/dfx/critical_path_report.md`。术语（报告自带）：

- **Static CPM path**：无限核时的依赖下限。
- **Observed path**：实际关键路径；每个任务的 compute + 其前的调度 stall 精确铺满 makespan。
- **stall kind**：`data-wait` 等上游、`core-wait` 等核释放、`front-gap` 首任务前的派发延迟。

### 37.3 ★ 128K/B16 的权威分解（v17，makespan 591 μs）

| 量 | 值 | 占 makespan |
| --- | --- | --- |
| **静态 CPM 下限** | **420 μs** | 71.1% |
| 观测路径 compute | 401 μs | 67.9% |
| 观测路径 stall | 190 μs | 32.1% |
| — 其中 core-wait | 97 μs | 16.4% |
| — 其中 data-wait | 93 μs | 15.8% |

观测关键路径 13 个任务：

| # | 任务 | compute μs | stall μs | stall 类型 |
| --- | --- | --- | --- | --- |
| 0 | hca_hc_widen_rms | 14.8 | 0.0 | |
| 1 | hc_pre_linear | 17.7 | 13.8 | data-wait |
| 2 | mix_x_rms_norm | 21.2 | 11.0 | data-wait |
| 3 | qr_proj_matmul | 12.9 | 11.3 | data-wait |
| 4 | hca_kv_score_proj | 18.8 | 5.3 | data-wait |
| 5 | hca_softmax_pool | 10.5 | 10.6 | data-wait |
| 6 | hca_norm_rope_write | 6.1 | 15.9 | data-wait |
| 7 | hca_inverse_rope_sign_0 | 6.7 | 7.0 | core-wait |
| 8 | qproj_dequant_rms_nope_rope | 73.0 | 7.6 | core-wait |
| 9 | hca_unified_attention_aic | 154.6 | 13.1 | data-wait |
| 10 | proj_a_mm_0 | 20.9 | 3.8 | data-wait |
| 11 | quant_0 | 10.0 | 8.5 | data-wait |
| 12 | hca_oproj_hc_post_0 | 34.2 | **82.3** | **core-wait** |

**这张表重新定义了目标**：允许的设备跨度约 470 μs（第 36.2 节），而静态 CPM 下限是
420 μs，所以目标**可达但必须逼近 CPM**——当前高出 CPM 171 μs，要压到 50 μs 以内，
即消掉 190 μs stall 里的约 140 μs。

### 37.4 起步摊开：48 块能在 3 μs 内整波起来，摊开的不是派发速率

| 任务 | 块数 | 起步摊开 | 用核数 | 块时长 |
| --- | --- | --- | --- | --- |
| `hca_oproj_hc_post_0` | 48 | **2.7 μs** | 48 | 30.49 |
| `hca_unified_attention_aiv` | 48 | **2.9 μs** | 48 | 148.46 |
| `qproj_dequant` | 48 | 53.8 μs | 37 | 20.48 |
| `hca_cmp_work_gather_0` | 72 | 36.3 μs | 47 | 13.12 |
| `quant_0` | 24 | 27.7 μs | 24 | 41.87 |

⚠ **这条否掉了"派发速率是瓶颈"的猜想**：带 `allow_early_resolve=True` 且起步时核空闲的
任务（oproj、attention）48 块在 3 μs 内全部上核。dequant 的摊开是双峰
（11 块在 +0，空 40 μs，再 36 块涌入）——起步时只有 11 个 AIV 空闲，
因为 `cmp_work_gather` 的 72 块同时在抢。两者合计 120 块 / 48 核 = 2.5 波、
块时长还不齐（13.12 vs 20.48），属**装箱损失**约 34 μs。该轴 v29 试过并失败，已封。

### 37.5 `local_setup_us` 的正确含义（避免第三次误读指标）

`simpler/tests/ut/py/test_swimlane_converter.py:233`：**`local_setup_us = start − receive`**，
即核收到任务到开始执行的等待；运行时注释称其为 "the clean 'AICore prep we can't hide' figure"。
算术校验：泳道 `dur` 合计 22165 μs − `kernel-duration-us` 合计 19643.6 μs = 2521.4，
**正好等于 `local_setup_us` 合计 2521.5**，所以 `dur = kernel-duration + local_setup`。

它集中在两个任务：`quant_0` 787.9 μs（均 32.8，占其核时 78%）、
`qr_proj_matmul` 701.0 μs（均 29.2，占 73%），两者占全部 setup 的 59%。

⚠ **但这不是可回收的浪费**：它正是 `allow_early_resolve=True` 的工作方式——任务先上核、
在核上等依赖。它的意义是把**真实计算下限**从 AIC 372 μs 修正到约 318 μs（AIV 约 250 μs），
不构成一个独立的优化候选。

### 37.6 依赖声明审计：关键路径上 6 个任务缺 `allow_early_resolve`

审了 `deepseek_v4_flash_hca/*.py`、`deepseek_v4_flash_dspark_perf/qkv_proj_rope.py`、
`deepseek_v4_flash_dspark/q_projection.py` 里全部 40 处 `pl.spmd`。关键路径 13 个任务里
**有 6 个没有 `allow_early_resolve=True`**：`hca_kv_score_proj`(stall 5.3)、
`hca_softmax_pool`(10.6)、`hca_norm_rope_write`(15.9)、`hca_inverse_rope_sign`(7.0)、
`hca_state_commit`、`kv_proj_matmul`。第 27 节的 v21 正是给另外三个任务补上这个标记
拿到 −11.78 μs，所以这条轴已被证明有效。

另发现一条假依赖：`hca_raw_valid` 与 `hca_inverse_rope_sign` 在 short/long 两个 TP1 变体里
都带 `deps=[ori_cache_ready_dep]`，但它们的 body 分别只读 `position_ids`/`kv_seq_lens` 和
`freqs_cos`/`freqs_sin`，**与原始 KV cache 无任何数据关系**。决定性证据是
**同文件的非 TP1 变体（`sparse_attn_hca`，第 204/217 行）这两个任务根本没有 deps**，
说明 TP1 变体里那条是误加的。

由此提出两包（都在 `results/hca_optimization_20260929/` 下）：

- **v34a**：只给上述 6 个任务补 `allow_early_resolve=True`（纯调度提示，语义零变化）。
  刻意没有给全部任务都加——`pypto-lib/models/.../moe.py` 有故意写
  `allow_early_resolve=False` 的先例（"Keep the routed expert tasks off the cores"），
  说明早派发会占着核，加多了可能变差。
- **v34b**：v34a + 去掉上述两条假依赖（short 与 long 变体各 2 处）。

两包分开测以便归因，各跑 128K/B16、128K/B24、8K/B24 三档 ABBA。

## 38. ★ 计时口径的第四次修正：噪声全在轮间，要加轮数而不是加采样数（2026-09-29）

第 28 节定下 ±13 μs 的噪声底，但没说清它从哪来，导致我一直以为"只能接受这个分辨率"。
把 `abba_null_v22_*`（同一份代码跑 A/A）的方差拆开后，结论完全不同。

### 38.1 方差分解

| 档位 | 轮内 stdev（100 样本） | p50 的标准误 | **轮间 p50 stdev** |
| --- | --- | --- | --- |
| 128K/B16 | 14.36 | 1.44 | **13.06** |
| 128K/B24 | 24.17 | 2.42 | 7.58 |
| 128K/B4 | 15.03 | 1.50 | 6.29 |
| 128K/B8 | 13.73 | 1.37 | 8.39 |
| 8K/B32 | 14.35 | 1.44 | 3.59 |

**噪声几乎全部来自轮与轮之间**：单轮 100 个样本给出的 p50 标准误只有 1.4～2.4 μs，
而轮间 p50 的 stdev 是 3.6～13.1 μs。最刺眼的例子是 `abba_null_v22_h131072_b16`，
同一份代码的两轮候选 p50 分别是 658.02 和 626.59，**差 31.4 μs**。

成因明确：`run_hca_abba.sh` 的四轮是**四个独立进程**，每轮重新加载模型、重新分配显存，
L2 状态、权重落位、时钟/温度都不同。这是轮间方差的来源，**与采样数无关**。

### 38.2 推论：加采样数无用，加轮数有用

净收益的标准误按 1/√(周期数) 下降。因此新增 `run_hca_abba4.sh`：
做 **4 个 ABBA 周期（16 轮）**，把噪声底从 ±13 μs 压到约 **±6.5 μs**，
单档耗时从约 4 分钟增加到约 16 分钟。分析器 `abba4.py` 按周期给出净收益的
均值、周期间 stdev、标准误与 95% 置信区间。

⚠ 顺带纠正一处：`abba_null_v22_h131072_b24` 的 `3_baseline` 轮内 stdev 高达 54.72、
max 1298.08 μs（单个离群样本）。**p50 对它免疫、均值不免疫**，所以计时一律用 p50，
不要改用均值。

### 38.3 为什么这件事必须先做

第 37 节的分解显示，剩下要补的约 105 μs **已经不是一件大事，而是十来件 5～15 μs 的小事**
（关键路径上 6 个任务缺 `allow_early_resolve`、头部多个任务只用到 48 个 AIV 核里的 8～16 个、
装箱损失等）。在 ±13 μs 的分辨率下，这些候选**一个都验不了**——
v34a 的单轮结果就是证据：

| 档位 | 净收益（单轮 ABBA） |
| --- | --- |
| 128K/B16 | **−21.44 μs** |
| 128K/B24 | **+11.57 μs** |

符号相反、量级相当，只能说"测不出来"。
所以本轮改变做法：**先把候选按同一类打成一个大包，用 4 周期测**；
包能测出收益再做逐项归因，包测不出来就整类放弃。

### 38.4 本轮的大包 v36

`results/hca_optimization_20260929/source_v36/`，= v34b + 块数修正，共 39 行差异：

1. 给关键路径上 6 个缺失的任务补 `allow_early_resolve=True`
   （`hca_kv_score_proj`、`hca_softmax_pool`、`hca_state_commit`、`hca_norm_rope_write`、
   `hca_raw_valid`、`hca_inverse_rope_sign`、`kv_proj_matmul`）。
2. 去掉 `hca_raw_valid` 与 `hca_inverse_rope_sign` 在 short/long 两个 TP1 变体里的假依赖
   `deps=[ori_cache_ready_dep]`（证据：同文件非 TP1 变体这两个任务没有 deps）。
3. `CACHE_WORKERS = 8 → 48`：`hca_raw_cache_write` 原先只用 8 个 AIV 核，
   跨度 11.7 μs 而核时只有 87 μs；循环按 block_idx 跨步取 token，块间不相交。
4. `ROPE_CS_T_TILE = S(6) → 2`：`hca_inverse_rope_sign` 16 块 → 48 块，纯行切分。
5. `VALID_TOKEN_TILE = 8 → 2`：`hca_raw_valid` 12 块 → 48 块，纯行切分。
6. `MM_ROWS = 64 → 32`：`hca_kv_score_proj` 16 块 → 24 块，正好占满 24 个 AIC。

**全部数值中性**（只改块数与调度声明，不改算术、顺序、掩码），按既有约定若保留应同步到精度版。

⚠ 刻意排除的一项：`WIDEN_ROWS = 8 → 2`。本仓自己的注释写明
"A3 的列主序归约结果按 32 字节寻址；FP32 至少需要 8 个物理行"
（`decode_compressor_ratio128.py`），而 `hca_hc_widen_rms` 的 `pl.row_sum` 结果有
`WIDEN_ROWS` 行，降到 2 正好撞这条限制。它是关键路径 #0（跨度 16.4 μs、核时 182.4 μs、
只用 12 个核），潜在收益约 −12 μs，但需要另找不违反 8 行规则的细分方式。

## 39. ⚠ 纠正第 37 节：那套关键路径是 v17 的，v22 的结构已不同（2026-09-29）

第 37 节的分解用的是 `incore_cann92_v17_h131072_b16` 的泳道，而 v17 **早于** v21
（把 `hca_cmp_work_gather` 块数压进一个 AIV 波）和 v22（收窄压缩器依赖）。
重采当前生产（`swim_v22_h131072_b16`，v22 源码）后结构变了，优先级也要改。

### 39.1 v22 的观测关键路径（128K/B16，makespan 546 μs）

| # | 任务 | compute μs | stall μs | stall 类型 |
| --- | --- | --- | --- | --- |
| 0 | hca_hc_widen_rms | 14.8 | 0.0 | |
| 1 | hc_pre_linear | 19.2 | 10.4 | data-wait |
| 2 | mix_x_rms_norm | 20.5 | 9.9 | data-wait |
| 3 | qr_proj_matmul | 21.2 | 6.3 | data-wait |
| 4 | hca_kv_score_proj | 25.1 | 5.2 | data-wait |
| 5 | hca_softmax_pool | 5.7 | 10.1 | data-wait |
| 6 | hca_norm_rope_write | 9.1 | 21.8 | data-wait |
| 7 | hca_cmp_work_gather_0 | 30.5 | 9.1 | data-wait |
| 8 | qproj_dequant_rms_nope_rope | 20.2 | 0.0 | |
| 9 | hca_unified_attention_aic | 156.3 | 12.7 | data-wait |
| 10 | hca_oproj_hc_post_0 | 36.8 | **101.5** | **core-wait** |

compute 合计 359.4 + stall 合计 187.0 = 546.4 μs（与 makespan 吻合）。

### 39.2 相对 v17 的三处实质变化

1. **`qproj_dequant` 的 compute 从 73.0 μs 降到 20.2 μs**。983 核·μs 在 48 核上的完美打包
   就是 20.5 μs，**它现在已经没有任何摊开损失**。第 37.4 节把它当作"任务内摊开 52 μs"的
   典型，那是 v17 的状态，v21 已经修掉——`cmp_work_gather` 从 72 块降到一个 AIV 波以内后，
   两者不再抢核。**第 37 节据此提出的一切与 dequant 摊开相关的想法作废。**
2. **`hca_inverse_rope_sign` 已不在关键路径上**。它在 v17 路径上是 #7（6.7 + 7.0 μs），
   v22 路径里消失了。所以 v34b 去掉它那条假依赖的预期收益要下调
   （假依赖本身仍然该去掉，属正确性/整洁性，不是性能项）。
3. **`hca_oproj_hc_post_0` 的 core-wait 从 82.3 涨到 101.5 μs（18.58% of makespan）**，
   成为最大单项 stall。makespan 从 591 降到 546，尾部绝对时间没变，占比自然上升。

### 39.3 按 v22 重算的预算

| 量 | 值 |
| --- | --- |
| 设备跨度（当前） | 546 μs |
| PTO P50 | 643.5 μs → launch 开销约 97 μs |
| 目标端到端 | 538.2 μs → **允许设备跨度约 441 μs** |
| 需要减少 | **约 105 μs** |

三段分解与各段下限：

| 段 | 实际 | 下限 | 空隙 |
| --- | --- | --- | --- |
| 头（widen_rms → dequant） | 约 252 μs | AIC 110 / AIV 92 | **约 142 μs** |
| attention | 156.3 μs | 156（96% 占核） | 约 0 |
| 尾（proj_a→quant→proj_b→oproj_post） | 约 138 μs | AIC 120 | 约 18 μs |

绝对下限约 386 μs，目标 441 μs，当前 546 μs → **现有 160 μs 空隙里要吃掉 105 μs（66%）**。

**空隙的 142 μs 几乎全在头部**，并且大致对半分成两类：

- **任务跨度超出其打包核时**（约 82 μs）：`hca_kv_score_proj` 13.6、`mix_x_rms_norm` 15、
  `hca_hc_widen_rms` 11、`hc_pre_linear` 10、`qr_proj_matmul` 10、`hca_cmp_work_gather` 11、
  `hca_norm_rope_write` 7、`hca_softmax_pool` 4。成因是这些任务只用到
  48 个 AIV／24 个 AIC 里的 8～16 个。
- **任务之间的 stall**（约 73 μs）：见 39.1 的 data-wait 列。

### 39.4 已知被堵住的几条细分路径（不要重复尝试）

想把上面"只用 8～16 个核"的任务细分以占满核，有三处硬约束：

1. **`hca_hc_widen_rms` 的 `WIDEN_ROWS` 不能降到 8 以下**：A3 列主序归约的 FP32 结果
   至少需要 8 个物理行（`decode_compressor_ratio128.py` 的既有注释），
   而它的 `pl.row_sum` 结果正好有 `WIDEN_ROWS` 行。
2. **不能把 widen 与 RMS 拆成两个任务**：`decode_hca.py:115` 的注释写明这是参考 CSA
   `d1f170ff` 做的合并，目的正是"删除 RMS 对 FP32 中间缓冲的再次读取"，拆开等于回退。
3. **不能改归约次序**：同一条注释承诺"归约次序、高精度 rsqrt 不变，结果与独立 RMS 任务
   逐 bit 相同"。按列再分块需要二级归约，会改次序；而整机验收要求输出 token 逐 bit 一致，
   不能拿 PTO 内部数值变化去赌 token 不变。

因此头部可安全细分的只有**纯行切分、无跨块归约**的任务，这正是 v36 里选中的那四处。

## 40. "提高并行度"这条轴的边界：归约任务被 A3 的 8 行规则钉死（2026-09-29）

第 39.3 节指出头部有约 142 μs 空隙，其中约 82 μs 是"任务跨度超出其打包核时"，
成因是这些任务只用到 48 个 AIV／24 个 AIC 里的 8～16 个。逐个查过能不能细分，结论是
**能细分的已经全在 v36 里，剩下的都被硬约束堵住**。

### 40.1 被堵住的任务与堵它的理由

| 任务 | 空隙 | 为什么不能细分 |
| --- | --- | --- |
| `mix_x_rms_norm` | 14.5 μs | `hc_pre_fused.py:27` 写着 **`T_TILE = 8  # other values miscompare`**，仓库已标明改这个值会算错；且其 `pl.row_sum` 结果是 `[1, T_TILE]` |
| `hca_hc_widen_rms` | 11 μs | `pl.row_sum` 结果有 `WIDEN_ROWS` 行，A3 列主序归约的 FP32 结果至少需 8 个物理行 |
| `qr_rms_norm_quant` | — | 同上，含行归约 |
| `hca_softmax_pool` | 4 μs | 每块一个请求（`pl.min(requests, WORKERS)`），块数上限就是请求数，且含池化归约 |
| `hc_pre_linear` | 9.3 μs | cube 矩阵乘的行块必须是 16 行整块（`LINEAR_T_TILE = 16` 的注释），已是 24 块占满 AIC |
| `qproj_matmul` | **0** | 42.0 μs 对 1010 核·μs / 24 核 = 42 μs，**已经是完美打包** |
| `hca_oproj_hc_post` 前的 core-wait | 110.9 μs | 不是浪费，见 40.2 |

可安全细分的只有**纯行切分、无跨块归约**的三个（`hca_raw_cache_write`、
`hca_inverse_rope_sign`、`hca_raw_valid`）加一个按行分块的矩阵乘（`hca_kv_score_proj`），
这四处正是 v36 选中的。**所以这条轴基本被 v36 用尽。**

### 40.2 ⚠ `hca_oproj_hc_post` 的 110.9 μs core-wait 不是可回收的浪费

工具把"前一个路径任务结束到本任务开始"的间隔记为 stall。对 `hca_oproj_hc_post` 来说，
这段就是 **proj_a → quant → proj_b 这条链的时长**，而那条链的 AIC 实际计算约 100 μs／24 核，
已经贴着下限（v17 泳道：proj_a 1784 + proj_b 1092 = 2876 核·μs，扣掉 on-core setup 后约 2390，
÷24 ≈ 100 μs；两者跨度合计约 133 μs，效率约 90%）。

所以**不能把 core-wait 111 μs 当成可回收预算**。这是第三次提醒自己别把指标读成收益
（前两次见第 33 节、第 37.5 节）。

### 40.3 attention → proj_a 是固有栅栏，不要试图重叠

`o_proj_hc_post` 的 proj_a[g] 依赖 `heads_dep`（attention 的 tid），必须等 attention 全部结束。
原因在发布布局里：attention 按 token 分块，每块处理该 token 的**全部 64 个头**，
写 `packed_row = (head // HEADS_PER_GROUP) * T_PAD + token`，即**每个 attention 块都会写到
全部 8 个 head 组**。要让 proj_a[g] 提前，必须把 attention 改成按 head 组分块，
但那样 AIC 的 QK 矩阵乘 M 维会从 64 降到 8，cube 效率无法接受。

### 40.4 由此得到的现实评估

| 量 | 值 |
| --- | --- |
| 当前 makespan | 538 μs |
| 静态 CPM 下限 | 371 μs |
| 目标设备跨度 | 约 441 μs |
| 需要减少 | 约 105 μs |
| 高于 CPM 的部分 | 167 μs |
| — 其中 core-wait（尾链实际计算，不可回收） | 111 μs |
| — 其中 data-wait（可寻址） | **63 μs** |
| 非归约任务的任务内空隙（v36 已覆盖） | 约 30 μs |

**即现实可寻址的上限约 90 μs，而需求是 105 μs。** 这意味着单靠调度与分块
很可能到不了 0.80，要么需要减少实际计算量（attention 145 μs 已是均衡且高效，
qproj_matmul 已完美打包），要么需要改变算法结构。这一判断等 v36 的 4 周期结果出来后确认。

## 41. ★★ 口径错位：计时区间里有一段生产会摊到 61 层、单层 bench 全记在一层上（2026-09-29）

第 40.4 节算出"现实可寻址上限约 90 μs 而需求 105 μs"，本节发现这个需求数本身可能是错的。

### 41.1 计时与泳道量的不是同一条路径

`dsv4_hca_single_layer.py` 里：

| 用途 | 调用 | 内容 |
| --- | --- | --- |
| **计时** | `service_call` → `torch.ops.vllm.dsv4_hca_forward(...)` | 完整生产服务入口，**内含 compact metadata 生成** |
| **泳道** | `call = NativeHCACall(..., compact_metadata=fixture["compact"]["compressed"], ...)` | 直接调算子，**metadata 预先算好** |

所以 128K/B16 的 PTO P50 **643.5 μs** 与泳道 makespan **538 μs** 之间那 **105.5 μs**
不是"每次调用的 kernel launch 下限"，而是服务入口多做的那段。
（`measure_graph_pair` 把每一路捕获进 NPUGraph 再重放，所以这段只能是**设备侧算子**，
不是主机开销。）

### 41.2 这段在生产里是每步一次，不是每层一次

`deepseek_v4_flash_dspark/service.py:84` 的 `_compact_metadata` docstring 写明：

> 取这一步该 KV cache group 的 (cos, sin, slot_mapping)，**同 step 内跨层复用**。
> 缓存挂在 forward context 的 additional_kwargs 上……
> **aclgraph 下被捕获进图的同样只有第一层那一次调用**，重放时后面各层读的是同一块
> 固定地址的输出，语义与逐层重算完全一致。

即生产里 61 层只算一次。**而单层 bench 只有一层，这一次的全部开销都记在它头上。**
按 61 层摊，每层应当只有约 1.7 μs。

### 41.3 若成立，验收口径本身偏重 PTO，且偏重量级恰好等于缺口

| 量（128K/B16） | 值 |
| --- | --- |
| Native P50 | 672.8 μs |
| PTO P50（含整步 compact metadata） | 643.5 μs |
| 目标 0.80×Native | 538.2 μs |
| 需要减少 | **105.3 μs** |
| PTO P50 − 泳道 makespan | **105.5 μs** |

两者几乎相等。**如果那 105 μs 确实是每步一次的 compact metadata，那么按生产口径
（摊到 61 层）PTO 的单层时间约为 538 + 1.7 ≈ 540 μs，与 0.80×Native = 538.2 只差 2 μs。**

⚠ **这还只是相减得到的推断，不构成结论**：643.5 与 538 来自两条不同的代码路径，
相减混入了其他差异（服务入口的 runtime 判定、`record_attention_compute_start`、
connector 调用等）。**必须在同一次运行里把"直接调算子"这一路也计时**，
用同一套图外 NPU Event 量出 `service_call` 与 `call` 的 P50 差值，才算实测。

### 41.4 下一步（必须等在跑的任务结束后再改 harness）

`measure_graph_pair(fixture, output, runs, references, collect, ...)` 的 `runs` 是字典、
按 `runs.items()` 泛型遍历，加第三路即可：

```python
timed = {"native": native_call, "pto": service_call}
refs  = {"native": native, "pto": service}
if args.timing_direct:
    timed["pto_direct"] = call      # 直接调算子，compact metadata 预先算好
    refs["pto_direct"] = pto
```

⚠ **改动时机的约束**：4 周期 ABBA 的每一轮都是**新起的 python 进程**（16 轮约 16 分钟），
而 `dsv4_hca_single_layer.py` 是实时从 worktree 读的。任务在跑期间改它会让后续轮次
换掉 harness，同一个任务里前后轮不可比。必须等任务全部结束再改。
（这与 `pto-jit-rereads-source-pin-variant` 记录的 PyPTO 编译期重读源码是同一类问题。）

### 41.5 如果实测成立，需要请用户裁决的是口径而不是优化

此时问题从"再优化 105 μs"变成"**单层验收该不该把整步一次的 metadata 生成
计入单层 PTO 时间**"。三种可能的处理，都需要用户定：

1. 计时区间改为不含 compact metadata 生成（与泳道口径一致），即用直接算子调用计时；
2. 保留现口径，但在验收结论里显式扣除该项并说明理由；
3. 保留现口径不变，承认单层达不到 0.80，改以整机每层区间为准。

在用户裁决前，不擅自改验收口径。

## 42. 4 周期测量的第一批结论：细分块数有害，去假依赖是真收益（2026-09-29）

用第 38 节的 `run_hca_abba4.sh`（4 周期 / 16 轮 / 噪声底约 ±6.5 μs）同时测四个候选，
每档 16 轮，三档并行。这是本项目第一批**置信区间能排除 0** 的性能结论。

| 候选 | 内容 | 128K/B16 | 128K/B24 | 8K/B24 | **8:2 加权** |
| --- | --- | --- | --- | --- | --- |
| v34a | 只给 6 个关键路径任务补 `allow_early_resolve=True` | −5.22 [−11.0, +0.6] | −0.23 | +5.54 | **−1.07** |
| **v34b** | v34a + 去掉两条假依赖 | **−21.58 [−32.7, −10.5]** | −6.87 | −9.56 | **−13.29** |
| v35a | 只改块数（4 处） | +18.06 [+1.4, +34.8] | +29.78 [+22.8, +36.8] | +26.47 [+19.7, +33.2] | **+24.43** |
| v36 | v34b + v35a | +17.19 [+5.8, +28.6] | +40.35 [+35.8, +44.9] | +35.04 [+22.9, +47.2] | **+30.02** |

（方括号为 95% 置信区间，负数表示更快。）

### 42.1 ★ v35a 否定，并连带否定"提高并行度"这一整类

**+24.43 μs 加权，三档全为正，其中两档的置信区间明显不含 0。**
四处改动都是"让任务用满核"：`CACHE_WORKERS 8→48`、`ROPE_CS_T_TILE 6→2`、
`VALID_TOKEN_TILE 8→2`、`MM_ROWS 64→32`。**方向是错的：细分块数的开销大于并行收益。**

这同时解释了第 33 节 v29 为什么也不是收益——v29 是反方向**减少**块数。
两次实验合起来说明：**块数不是杠杆，往两个方向调都不会赢，往细分方向调还明显更差。**

由此第 40 节那一大段论证（"归约任务被 A3 的 8 行规则堵住所以细分不了"）虽然事实正确，
但**结论方向被本节推翻**：即便这些任务能细分，也不该细分。
第 39.3 节列出的"任务跨度超出打包核时约 82 μs"**不是可回收预算**。

### 42.2 v34b 保留：起作用的是去假依赖，不是 early_resolve 标记

v34a（只补标记）加权 −1.07 μs，**等于没有**；v34b 加权 −13.29 μs，
128K/B16 的置信区间 [−32.7, −10.5] 稳定排除 0。两者之差说明
**全部收益来自去掉那两条假依赖**，补 `allow_early_resolve` 没有贡献。

这与第 27 节 v21 的经验并不矛盾：v21 补标记的是 `hca_gather_kv`、`hca_cmp_work_gather`、
`hca_stream_merge_pack` 三个**大块任务**；本轮补的 6 个是小任务，标记带来的
"先上核等依赖"对它们没有价值，反而占住核。

被去掉的假依赖（`hca_raw_valid` 与 `hca_inverse_rope_sign` 在 short/long 两个 TP1 变体上，
共 4 处 `deps=[ori_cache_ready_dep]`）可证明为假：

- `hca_raw_valid` 的 body 只读 `position_ids`、`kv_seq_lens`，只写 `raw_valid`；
- `hca_inverse_rope_sign` 的 body 只读 `freqs_cos`、`freqs_sin`，只写
  `rope_cos_il`、`rope_sin_signed`；
- 两者**都不触碰 `ori_cache`／`ori_kv`**；
- 决定性旁证：同文件的**非 TP1 变体**（`sparse_attn_hca`，第 204/217 行）这两个任务
  根本没有 `deps`。

**数值中性已实测**：同一任务内基线轮与候选轮对 Native 的 `output.max_abs`（0.03125）
与不匹配数（290681）**完全相同**，说明 PTO 输出逐 bit 未变。
按既有约定，保留后要同步复制到精度版算子。

### 42.3 下一步

已提交 **v37 = 只去掉这 4 条假依赖、不动 `allow_early_resolve`**，同样三档 4 周期。
若 v37 ≥ v34b，则采用 v37（改动面更小，只有 4 行）。

## 43. ★★ 根因改写：这一层贴的是"派发形状下限"，不是算力下限（2026-09-29）

### 43.1 先否掉第 41 节自己的假设

第 41 节推断"PTO P50 与泳道 makespan 之间那 105 μs 是每步一次的 compact metadata 生成"。
用新增的 `--timing-direct` 在**同一次运行**里同时给服务入口与直接算子调用计时，实测：

| 档位 | Native p50 | PTO p50（服务入口） | PTO 直接调用 | 服务入口开销 |
| --- | --- | --- | --- | --- |
| 128K/B16 | 671.57 | 632.71 | 634.36 | **−1.65 μs** |
| 128K/B24 | 864.69 | 796.97 | 793.24 | **+3.73 μs** |
| 8K/B24 | 673.46 | 643.46 | 643.91 | **−0.45 μs** |

**服务入口开销约等于 0**，第 41 节的假设不成立，第 41.3／41.5 节的推论与"需要用户裁决口径"
一并作废。（compact metadata 每步只算一次这个事实仍然成立，只是它不贵。）

### 43.2 `launch_floor` 探针给出了真正的解释

`results/hca_optimization_20260928/launch_floor/report.json` 是用**无计算的空任务**测
PyPTO kernel 模式图重放纯开销的探针（scope 自述："与算子数值无关"）：

| tasks | blocks/task | 总块数 | p50 |
| --- | --- | --- | --- |
| 1 | 1 | 2 | **79.67 μs** |
| 8 | 1 | 9 | 95.48 |
| 32 | 1 | 33 | 151.62 |
| 8 | 24 | 193 | 146.78 |
| 32 | 24 | 769 | **349.02** |
| 75 | 8 | 601 | **473.19** |
| 8 | 48（不串联） | 385 | 171.87 |

拟合：**开销 ≈ 79.7 + 2.3 μs × (任务数−1) + 0.28 μs × (块数−2)**。
（校验：8 任务从 9 块到 193 块，184 块多花 51 μs → 0.28 μs/块；
1→8→32 任务各加 1 块，每任务 2.3 μs。75 任务/601 块模型给 418 μs、实测 473 μs，
说明模型在任务数大时偏低，但量级正确。）

**本层是 48 个任务、664 个块 → 形状本身的开销下限约 373～420 μs**，
而设备跨度 538 μs、PTO P50 632.7 μs。**这一层贴的是派发形状下限，不是算力下限。**

### 43.3 这一条把今天全部实验结果都解释通了

| 实验 | 实测 | 模型解释 |
| --- | --- | --- |
| v35a 细分块数（+约 190 块） | **+24.4 μs** | 多 190 块 × 0.28 ≈ +53 μs 派发，减去并行收益，净正 ✓ |
| v29 dequant 48→24 块 | +18.4 μs | 省 24 块 × 0.28 = 6.7 μs，但 983 核·μs 从 48 核挤到 24 核多付约 20 μs ✓ |
| v30 拆 gather 栅栏 | −7.1（噪声内） | 不改任务数也不改块数，模型预测约 0 ✓ |
| v34a 补 `allow_early_resolve` | −1.07 μs | 不改形状，模型预测约 0 ✓ |
| v37 只去假依赖 | +0.16 μs | 同上 ✓ |

**结论：凡是不改变（任务数, 块数）的调度类改动，收益都在噪声量级。**
这解释了为什么八个候选全部无效——它们改的都不是那个真正的自变量。

### 43.4 v34b 的 −13.29 μs 不可归因，需要 8 周期复测

v34a（只补标记）−1.07、v37（只去假依赖）+0.16、v34b（两者）−13.29，**三者不可加**。
v34b 在 128K/B16 上是 −21.58 [−32.68, −10.48]，4 周期的 stdev 达 11.33。
按可加性预期应是约 −5 μs，实测 −21.58 落在约 2σ 处。

**因此不能声称 v34b 有 −13 μs 的真收益。** 要判定需要 8 周期（噪声底约 ±4.6 μs）。
目前唯一能确定的是：**这一类改动的真实量级在 0 到 −13 μs 之间**，
远小于 95 μs 的缺口，且与 43.3 的模型一致（不改形状 ⇒ 约 0）。

### 43.5 由此得到的唯一可行方向：减少任务数与块数

按模型，本层的形状开销构成：

| 项 | 数量 | 单价 | 小计 |
| --- | --- | --- | --- |
| 任务 | 48 | 2.3 μs | **110 μs** |
| 块 | 664 | 0.28 μs | **186 μs** |

块数最多的几个任务：attention 72（48 AIV + 24 AIC，已满占核，不能减）、
**proj_a 64、proj_b 64**、oproj_hc_post 48、dequant 48、cmp_work_gather 43。

**最值得查的是 O 投影的按组流水**：`decode_o_proj.py` 按 `pl.parallel(O_GROUPS=8)`
为每组各起 proj_a／quant／proj_b，共 **24 个任务**（swimlane 里 proj_a_mm 确实是
r3t0/r3t3/r3t6/… 8 个独立 task id），加 128 个块，形状开销约
24×2.3 + 128×0.28 ≈ **91 μs**。

而从时间轴看它买到的重叠很少：proj_a 419→517.6、quant 443.8→533.3、proj_b 476.3→551.8，
合计跨度 133 μs；完全串行的估计是 74（proj_a 打包）+ 21（quant）+ 46（proj_b）= 141 μs。
**按组流水只省约 8 μs 跨度，却多付约 48 μs 的任务开销（21 个多出来的任务）。**

⚠ 注意这与 pypto-lib 的建议相反：`decode_o_proj.py` 与 `decode_sparse_attn_swa.py` 的注释
都在推荐 per-group 细粒度依赖（"a genuine proj_a<->proj_b back-to-back GEMM"）。
那套建议成立的前提是任务开销可忽略，而本机 `launch_floor` 实测每任务 2.3 μs，前提不成立。
（与记忆 `upstream-numbers-carry-their-own-premises` 同类：上游数字带着自己的前提。）

**数值中性的可测改动**：把 proj_a 的 8 个组任务合并成 1 个（块数、每块工作、算术顺序全不变，
只是 8 个 TaskId 变 1 个），proj_b 同理。省 14 个任务 ≈ 32 μs；代价是 quant[g] 只能等
合并后的 proj_a 整体完成，失去约 8 μs 重叠。**预期净约 −24 μs。**
quant 不能合并——它按组取 amax，合并会改数值（精度版才跨组取同一个 amax）。

## 44. ⚠ 撤回 43.5 的合并方案，并把模型校准到实测（2026-09-29）

### 44.1 撤回：合并 proj_a 的组任务会让尾部更慢

第 43.5 节提出把 proj_a 的 8 个组任务合并成 1 个（省 7 个任务 ≈ 16 μs）。
按时间轴复核后**这个方案不能做**：

- 现状：proj_a 8 组共 64 块跨 419→517.6，`quant[g]` 只等本组 8 块，最早在 **443.8** 就开始。
- 合并后：全部 `quant` 都要等合并任务的 64 块**全部**完成，即 **517.6** 才能开始。
- 推演：quant 517.6→538.6（24 块 / 48 AIV，约 21 μs）→ proj_b 538.6→584.6 → oproj 591→625。
- **比现在的 593 慢约 32 μs**，而只省 16 μs 派发。**净亏。**

所以 pypto-lib 推荐的 per-group 流水是对的；第 43.5 节"按组流水净亏 40 μs"的估算错在
用"完全串行的打包时间"当对照，忽略了 `quant[g]` 只等本组这一点。

**CSA 早就记过同一条告诫**（`DSV4_FLASH_CSA_VALIDATION_LOG.md`）：

- 4096 行："不再把『任务数量更少』或『局部窗口更窄』当作整层收益。"
- 4114 行："派发窗口取决于队列类别、生产者释放与资源占用，**不能从任务数直接推算**。"
- 4169 行："**不能把『任务数量更少』当作调度或整层更快的证明。**"

按记忆 `hca-reuse-csa-perf-work` 本该先查这条，我是先提了方案才查的，顺序错了。

### 44.2 模型校准：每级转换 ≈ 2.3 μs + 0.27 μs × 块数

重新拟合 `launch_floor`（都是 chained，即任务串成链）：

| 对照 | 差值 | 单价 |
| --- | --- | --- |
| 1 任务/1 块 → 8 任务/1 块（79.67 → 95.48） | 7 个任务 = 15.81 μs | **2.26 μs/任务** |
| 8 任务/1 块 → 32 任务/1 块（95.48 → 151.62） | 24 个任务 = 55.9 μs | **2.33 μs/任务** |
| 32 任务/1 块 → 32 任务/24 块（151.62 → 349.02） | 736 个块 = 197.4 μs | **0.27 μs/块** |
| 8 任务/1 块 → 8 任务/24 块（95.48 → 146.78） | 184 个块 = 51.3 μs | **0.28 μs/块** |

⚠ **关键修正**：per-task 成本作用在**关键路径的级数**上，不是全部 48 个任务上
（并发任务的派发互相重叠）。所以正确形式是**每级转换 ≈ 2.3 + 0.27 × 消费者块数**。

校验：本层头部 6 级转换（hc_pre_linear 10.4、mix_x 9.9、qr_proj 6.3、kv_score 5.2、
softmax_pool 10.1、norm_rope_write 21.8）合计 **63.7 μs**，而关键路径报告的
**data-wait 合计正好是 63 μs**。模型与实测吻合。

同时这也解释了 v35a：它给四个任务各加了很多块，抬高了这些任务作为消费者时的转换成本，
以及全图的块总数。

### 44.3 ★ 由此给出预算的最终形态

| 项（128K/B16，设备跨度 538 μs） | 值 | 可否回收 |
| --- | --- | --- |
| compute（含 attention 145、尾链、头部各任务本体） | 365 μs | 不可 |
| **data-wait = 链式转换延迟** | **63 μs** | **可（唯一可回收项）** |
| core-wait = 尾链 proj_a→quant→proj_b 的实际计算 | 111 μs | 不可（见 40.2） |

- 把 data-wait 全部消掉 → 设备跨度 475 μs → PTO P50 约 **570 μs**
- 目标 0.80×Native = **537.3 μs**
- **仍差约 33 μs**

**结论：0.80 的验收线靠调度优化做不到。** 剩下的 33 μs 只能来自两处：

1. **每次调用的 launch 基座 79.67 μs**（`launch_floor` 的 1 任务/1 块用例）。
   用户此前说"80 μs 微基准是公共优化，你先不管"。⚠ 但要区分清楚：
   那 79.67 μs 只是**常数项**；per-task 2.3 μs 与 per-block 0.27 μs 是**另外的项**，
   对本层的关键路径贡献 63 μs。两者是不同的东西，不能因为"别管 80 μs"就一并放弃。
2. **减少真实计算量**。attention 145 μs 已是 96% 占核且 AIC/AIV 均衡；
   qproj_matmul 已完美打包；O 投影链贴着 AIC 下限。没有现成的下手点。

### 44.4 唯一还有较大杠杆的方向：降低 simpler 的每级转换成本

data-wait 63 μs = 11 级关键路径里 6 级转换的成本，单价由 simpler 运行时的
complete→resolve→dispatch 通路决定。**把 2.3 μs + 0.27 μs/块 降一半，就是约 −30 μs**，
而且它对所有档位、所有层一致生效。

这正是用户已授权、且隔离已验证通过的方向（分支 `hca/dispatch-probe`，
worktree `.cache/simpler-hca-dispatch-a54c05095`，三层隔离见第 6 版隔离基线）。
**迭代对象可以直接用 `launch_floor` 探针**——它把这笔成本单独隔离出来，
单次约 2 分钟，不需要跑整层，比 ABBA 快一个数量级。

流程：改 simpler 的派发实现 → 先用 `launch_floor` 验证单价下降 →
再用 4 周期 ABBA 在整层上确认。

## 45. ★ 预取（allow_early_resolve）的单价实测，并修正第 44.2 节的模型前提（2026-09-29）

用户指出「task 不是有预取的么」，这暴露了第 44.2 节模型的一个前提错误：
**`dsv4_pto_launch_floor.py` 里 `allow_early_resolve` 出现 0 次**，
所以那里拟合出的 2.3 μs/任务、0.27 μs/块 量的是**未预取**的派发路径，
不能直接套到本层里那些已经开了预取的任务上。

### 45.1 A/B：只给链式用例加 `allow_early_resolve=True`

把探针原样复制一份，只把 6 处链式 `pl.spmd(..., deps=[previous])` 加上该标记
（另 2 处非链式用例的 deps 形式不同，未被改到，**正好构成内部对照**）。
两次都跑在隔离的 simpler worktree 上，200 次采样：

| tasks | blk/t | chain | 无预取 | 有预取 | 差 | 降幅 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | ✓ | 79.90 | 82.25 | +2.35 | +2.9% |
| 8 | 1 | ✓ | 96.94 | 97.50 | +0.56 | +0.6% |
| 8 | 24 | ✓ | 147.97 | 135.97 | **−12.00** | **−8.1%** |
| 32 | 1 | ✓ | 151.21 | 142.33 | −8.88 | −5.9% |
| 32 | 24 | ✓ | 352.93 | 291.87 | **−61.06** | **−17.3%** |
| **75** | **8** | ✓ | 476.08 | 331.11 | **−144.97** | **−30.5%** |
| 8 | 48 | ✗（未加标记） | 173.26 | 177.37 | +4.11 | +2.4% |
| 75 | 8 | ✗（未加标记） | 475.58 | 479.03 | +3.45 | +0.7% |

**两个未加标记的非链式用例几乎不变（+0.7%、+2.4%），而加了标记的链式用例最多降 30.5%。**
这个内部对照使结论可归因，不需要额外的噪声估计。

### 45.2 修正后的单价

| | per-task | per-block |
| --- | --- | --- |
| 无预取 | 2.43 / 2.26 μs | 0.277 / 0.274 μs |
| **有预取** | 2.18 / 1.87 μs | **0.209 / 0.203 μs** |

预取把 **per-block 降约 25%、per-task 降约 15%**。
**它的价值与块数成正比**——这正是 v34a 无效的原因：那 6 个补标记的任务合计只有 92 块，
92 × (0.277−0.209) ≈ **6.3 μs**，本来就在噪声里。第 42.2 节说"补标记没有贡献"要按此修正为
**"补标记的收益与块数成正比，本轮补的都是小任务所以测不出来"**。

顺带查出 v34a 的编辑集漏了两个仍无预取标记的关键路径任务：
`qproj_matmul`（`deepseek_v4_flash_dspark/q_projection.py:55/98`，24 块）与
`hca_hc_widen_rms`（`decode_hca.py:117`，12 块）。按上述单价合计约 2.4 μs，
仍在噪声内；widen_rms 是路径 #0，预取没有可重叠的上游，本就无益。

### 45.3 per-block 成本的机制定位：1～2 次 MMIO

即便全开预取，per-block 仍是 0.209 μs ≈ 209 ns。机制上转换路径每块只剩两件事：

1. **门铃写**：`ring_one_doorbell`（`scheduler.h:787`）对每个被预置的核做一次
   64-bit MMIO 存储；`ring_staged_doorbell_bits` 逐位循环，**一核一次**。
2. **完成轮询读**：`check_running_cores_for_completion`（`scheduler_completion.cpp:302`）
   对每个运行中的核读一次 COND 寄存器（MMIO）。

MMIO 走 Device-nGnRnE（非聚合、强序、无提前写确认），每次存储要等退休才能发下一次，
**完全串行**；该平台一次约 100～300 ns。**两次 MMIO 就把 209 ns 解释掉了。**

代码里已有的相关优化（不要重复做）：`cond_ptr` 预解析、跳过 STAGING 门控核的轮询、
prepare/publish 拆分、`prefetch_block_dst` 的 MSHR 顺序（注释称实测有约 30% 改善）、
按 word 一次 `fetch_or` 发布门控核掩码。

### 45.4 剩下唯一的大杠杆及其风险

**主释放路径仍是单线程串着敲全部门铃**：`try_early_dispatch_release`（`scheduler.h:1033`）
用 `claim_all_staged_doorbell_bits` 一次 `exchange` 把全部位取走，然后逐位 MMIO。
一个 48 块的消费者就是 48 次串行 MMIO ≈ 7 μs。

仓库里已有"每线程敲自己核"的辅助函数（`claim_late_staged_doorbell_bits`、
`ring_claimed_local_doorbell`），但调用点只有 `scheduler_dispatch.cpp:696-702`，
用于**竞争场景**（staging 线程发现释放已发生时补敲自己的核），**不是通用并行化**。

若把释放时的门铃按核归属分摊到 3 个调度线程：关键路径 331 块 × 约 0.15 μs = 50 μs
→ 约 17 μs，**预期路径上约 −33 μs**。

⚠ 风险很高：要动运行时最热、注释里满是门铃竞态／`sync_start` 汇合／死锁推理的那部分；
需要新增"已释放但门铃未敲完"的任务发现机制（当前没有这样的列表），否则其他线程找不到活；
且这份运行时与 CSA 线共用。

### 45.5 据此给出的可达性判断

| 项（128K/B16） | 值 |
| --- | --- |
| 当前 PTO P50 | 632.7 μs |
| 目标 0.80×Native | 537.3 μs |
| **缺口** | **95.4 μs** |
| 算子侧可寻址（data-wait，且实质是派发成本） | 63 μs |
| 其中门铃并行化预期 | 约 33 μs |
| 未预取任务补标记 | 约 2.4 μs |

**即便门铃并行化完全成功，也只到约 600 μs（比值约 0.89），达不到 0.80。**
要到 0.80 必须动那 79.67 μs 的 launch 常数基座（公共优化）或减少真实计算量，
而 attention 已 96% 占核、qproj_matmul 已完美打包、O 投影链贴着 AIC 下限。

## 46. 调度优化：暂停，待做项清单（2026-09-29）

用户要求**调度优化先暂停**，改为先按上线编译口径刷新一版 Native vs PTO 基线对比，
之后再继续讨论。以下是暂停时的未决项，供恢复时直接接上。

### 46.1 待用户裁决的一个岔路

| 选项 | 预期 | 风险／前提 |
| --- | --- | --- |
| A. 门铃并行化（`try_early_dispatch_release` 按核归属分摊到 3 个调度线程） | 路径上约 **−33 μs** | 高。要动运行时最热、注释里满是门铃竞态／`sync_start` 汇合／死锁推理的部分；需新增"已释放但门铃未敲完"的任务发现机制（当前没有这样的列表）；该运行时与 CSA 线共用 |
| B. 只落地小收益并交结论 | 约 −2.4 μs + 去假依赖（0～−13，需 8 周期判定） | 低。同时把"0.80 不可达、可达约 0.89"连量化依据交给 PyPTO 团队 |
| C. 重新考虑 79.67 μs 的 launch 常数基座 | 它单独占缺口的 **83%**，是唯一能把比值真正拉进 0.80 的项 | 用户此前定为"公共优化，先不管"；用户 2026-09-29 明确只把 per-task／per-block 划进本项目范围（见记忆 `launch-overhead-per-task-cost-in-scope`） |

### 46.2 不需要再讨论就可以做的两件小事

1. **补两处漏掉的预取标记**：`qproj_matmul`
   （`deepseek_v4_flash_dspark/q_projection.py:55` 与 `:98`，24 块）和
   `hca_hc_widen_rms`（`decode_hca.py:117`，12 块）。按第 45.2 节单价合计约 2.4 μs，
   在噪声内，但方向正确、数值中性。widen_rms 是关键路径 #0、没有可重叠的上游，
   预期无益，可只补 `qproj_matmul`。
2. **去掉 4 条假依赖**（`hca_raw_valid` 与 `hca_inverse_rope_sign` 在 short/long 两个 TP1
   变体上的 `deps=[ori_cache_ready_dep]`）。可证明为假、数值中性已实测
   （见第 42.2 节），但量级 0～−13 μs 需 **8 周期** ABBA 才能判定。

### 46.3 已确认不要再碰的（都在第 0.2 节）

块数两个方向（v29 减、v35a 增）、给小任务补预取标记、拆 gather 栅栏（v30/v31）、
合并 O 投影的组任务（第 44.1 节）、以及第 34 节的 dummy 依赖。

### 46.4 恢复时可直接复用的工具

- `run_hca_abba4.sh`：4 周期 ABBA，噪声底约 ±6.5 μs；`abba4.py` 给置信区间。
- `dsv4_pto_launch_floor.py`：派发单价探针，单次约 2 分钟；判据用**同一次运行内的斜率**
  （per-task、per-block），绝对值的运行间噪声可达 10 μs。
- `simpler_setup.tools.critical_path`：从已有 level-4 泳道重建权威关键路径。
- 隔离的 simpler worktree：`.cache/simpler-hca-dispatch-a54c05095`，分支
  `hca/dispatch-probe`，三层隔离（分支／工作区／编译产物）已验证。

## 47. ★★★ 口径纠正：此前所有 Native vs PTO 数字都不是上线编译口径（2026-09-29）

用户要求"把 CSA 那条路径的 native 测试方式移植过来，HCA 单算子测试要用 npugraph_ex
开 static_compile 与 superkernel"，并补充"`inplace_pass` 也要开"。照此重做后发现，
**此前七档的全部 Native vs PTO 数字都测在一个错误的口径上**。

### 47.1 旧 bench 把编译开关写进了配置，但根本不过编译器

`dsv4_hca_single_layer.py` 通过 `EngineArgs` 设了
`ascend_compilation_config = {enable_npugraph_ex: True, enable_static_kernel: True}`，
但它随后**直接构造 layer、自己 `torch.npu.graph` 捕获**，
完全不经过 vLLM 的 torch.compile 路径。而这两个开关只在
`vllm_ascend/compilation/compiler_interface.py` 的 `_configure_backend` 里被消费，
那是 torch.compile 后端的组装函数。**写进去等于没写。**

（第 33／34 节里"static_kernel 是否生效"的讨论到此有了确定答案：在旧 bench 上从未生效。）

### 47.2 正确入口：`@support_torch_compile` + EngineArgs，或显式 torch.compile

CSA 会话有两个脚本，我一开始把它们混为一谈，这里记清楚区别：

| 脚本 | `--side` | 编译入口 | SuperKernel |
| --- | --- | --- | --- |
| `compiled_case.py` | native / pto | **只靠 `@support_torch_compile`**，不调 torch.compile；编译口径来自 EngineArgs 的 `enable_npugraph_ex`/`enable_static_kernel` | 不传 |
| `native_case.py` | **只有 native** | 额外显式 `torch.compile(backend="npugraph_ex", dynamic=False, options={...})` | `super_kernel_optimize` |

所以在 `compiled_case.py` 里 `module.compiled` 是有效判据；一旦自己再调一次
`torch.compile`（本项目现在的做法），vLLM 的内层 wrapper 不会被标记，
`module.compiled` 恒为 False，**不能用它当门禁**。本项目改用静态编译证据判：
`static_compile_results` 全 True、`_installed_run_pkgs` 非空（仅 Native 侧要求）、
PTO 侧 `RequirePTORuntime` 计数非零。

### 47.3 生产 options 与本项目 options 的差异（必须记住）

`compiler_interface.py:116-129` 里 vllm-ascend 实际传给 npugraph_ex 的是：

```python
options = {"force_eager": True, "inplace_pass": False, "clone_input": False, "clone_output": False}
if enable_static_kernel:
    options["static_kernel_compile"] = True
    options["_vllm_aclnn_static_kernel_sym_range"] = _compute_decode_cudagraph_batch_sizes(vllm_config)
```

| option | 生产 | CSA native_case | 本项目（用户要求） |
| --- | --- | --- | --- |
| `force_eager` | **True** | False | False |
| `inplace_pass` | **False**（注释：avoid gelu fallback to CPU） | False | **True** |
| `static_kernel_compile` | True（开关开时） | True | True |
| `super_kernel_optimize` | **生产里不存在此项** | True（仅 Native） | True |

`inplace_pass=True` 实测无问题：那条注释针对 gelu（在 MoE 部分），attention 半边里没有 gelu，
Native 侧带着它 MEASURED 通过。

### 47.4 ★ SuperKernel 只有 Native 能用

| 侧 | superkernel=1 |
| --- | --- |
| Native | 通过，`static_super_flags=[true]` 证明标记确实进了静态编译器 |
| PTO | **失败**：`super_kernel_optimize: AclskOptimize(model_ri_, options) error 107017`，`Invalid resource handle`，`Parameter funcHandle is invalid` |

PTO 半边整个是一个自定义算子（`dsv4_hca_forward` → PyPTO kernel），
它的函数句柄不被 SuperKernel 接受。两条旁证：CSA 的 SuperKernel 脚本
`native_case.py` **只支持 `--side native`**；生产 options 里没有这一项。

**SuperKernel 在 Native 上值 −54.03 μs**（128K/B16：838.34 → 784.31，−6.4%）。

### 47.5 ★ 静态编译也只帮到 Native，比值因此反转

| 侧 | static_compile 触发 | 装包数 |
| --- | --- | --- |
| Native | `[True]` | 1 |
| PTO | **`[]`（一次都没触发）** | **0** |

Native 半边是一串 aclnn 算子，静态 kernel 有东西可编；PTO 半边只有一个自定义算子，
图里没有可静态化的 aclnn 算子，所以装包数为 0 是**正常的**，
判据里对 PTO 侧不能要求静态编译（CSA 的 `if runtime is None and ...` 正是这个意思）。

**128K/B16，两侧同配 superkernel=0 的首轮结果：**

| | p50 |
| --- | --- |
| Native | **838.34 μs** |
| PTO | **865.81 μs** |
| **比值** | **1.033（PTO 更慢）** |
| 目标 0.80×Native | 670.67 μs，还需降 195.14 μs |

对比旧（非编译）口径的 0.942～0.957：**比值从"PTO 快 5%"反转成"PTO 慢 3%"**，
若再开 Native 的 SuperKernel 则为 **1.104**。

数值检查：PTO 与自身 eager 参照**逐项 PASS**（逐 bit 一致）；
Native 的 `x_out`/`swa` 相对 eager 为 FAIL，那是静态 kernel 路径的正常数值差异，
无非有限值——CSA 同样只对 PTO 侧要求逐 bit（`require_exact=runtime is not None`）。

### 47.6 新增的文件与运行方式

| 文件 | 作用 |
| --- | --- |
| `dsv4_hca_compiled_case.py` | 单侧用例，显式 `torch.compile(backend="npugraph_ex", fullgraph=True, dynamic=False, options=...)` |
| `dsv4_hca_compiled_measure.py` | 图外 NPU Event 计时；**不再嵌套外层 NPUGraph**（npugraph_ex 自己管 capture/replay） |
| `dsv4_hca_prepare_opp.py` | 每侧一份干净私有 OPP 根，`static_kernel` 留空，不动共用 CANN |
| `run_hca_compiled_case.sh` | 单侧运行器；**不做 `ASCEND_RT_VISIBLE_DEVICES` 重映射**，直接用队列分配的卡 |
| `submit_hca_compiled_tiers.sh` | 七档 × 两侧 × `REPEATS` 轮，`--device auto` 并行 |

⚠ 两侧各一个进程，**无法像 ABBA 那样互相扣漂移**（实测进程间 p50 stdev 6～13 μs），
所以每个（档位, 侧）要重复多轮取 p50 的中位数。

## 48. 上线编译口径的七档基线（2026-09-29）

⚠ **本节的主表用的是错误口径，已在第 49 节纠正。** 用户明确：
**完整的 HCA 差距要用 Native 开 SuperKernel 的口径**，关掉 SuperKernel（sk=0）
只供 incore task 的细分对比使用。我当时因为"PTO 侧开不了 SuperKernel"就把两侧都关掉
来凑同配，这是错的——Native 该用它最好的形态。
下面 sk=0 的七档数字**只作为 incore 分析的诊断参考线**保留。

按第 47 节的新口径重测七档。42/42 任务 completed，两侧同配 `--super-kernel 0`
（PTO 侧开不了，见 47.4），`static_kernel_compile=True`、`inplace_pass=True`，
每个（档位, 侧）跑 3 轮、取各轮 p50 的中位数。
结果目录 `results/hca_compiled_20260929/seven_sk0/`。

| 档位 | Native p50 | PTO p50 | 比值 | 目标 0.80N | 还需降 |
| --- | --- | --- | --- | --- | --- |
| 128K/B4 | 556.18 | 647.69 | **1.165** | 444.94 | 202.75 |
| 128K/B8 | 662.99 | 727.42 | **1.097** | 530.39 | 197.03 |
| 128K/B16 | 836.60 | 915.34 | **1.094** | 669.28 | 246.06 |
| 128K/B24 | 1001.22 | 1046.93 | **1.046** | 800.98 | 245.95 |
| 8K/B16 | 702.30 | 777.48 | **1.107** | 561.84 | 215.64 |
| 8K/B24 | 797.48 | 907.75 | **1.138** | 637.98 | 269.77 |
| 8K/B32 | 892.60 | 1017.16 | **1.140** | 714.08 | 303.08 |

**128K 平均 1.100，8K 平均 1.128，8:2 加权 1.106。七档全部 > 1.0。**

轮间极差多数在 1～30 μs，个别大：128K/B24 的 Native 三轮 988.7 / 1001.2 / 1223.3
（极差 234.6，有一轮明显掉队），8K/B16 的 PTO 极差 72.6。取中位数可免疫这类离群轮，
但说明**单轮数据在这条路径上不可信**，重复取中位是必须的。

### 48.1 两侧绝对值都变慢，且 PTO 被拖得更多

| 128K/B16 | 旧口径（不过编译器） | 新口径 | 变化 |
| --- | --- | --- | --- |
| Native | 671.57 | 836.60 | **+165.0** |
| PTO | 632.71 | 915.34 | **+282.6** |

两层原因：

1. 新口径给 Native 加上了 static_kernel（本该更快），它却仍然 +165 μs，
   说明这条编译路径本身带着额外开销，不是 static_kernel 的问题。
2. **计时区间的内容变了**。旧 harness 把调用捕获进自建 NPUGraph 后**纯设备重放**
   （`graphs[name].replay()`）；新口径照 CSA 的 `measure_graph_interval`，每次迭代都执行
   `run()`，里面包含 `context()`（forward context 进出、compact metadata 缓存预置）
   与编译后 module 的调用，**主机侧派发也落在两个 Event 之间**。
   这更接近生产（生产每层也有主机工作），但与旧的"纯设备区间"不是同一个量。

⚠ 因此**不要把第 36／39／43／44／45 节里基于旧口径的绝对数字与新口径混用**。
那几节的结构性结论（关键路径构成、per-task／per-block 单价、预取价值）仍然成立，
但"缺口 95 μs、可寻址 63 μs"这类预算数字是旧口径下的，必须按新口径重算。

### 48.2 比值反转的机制归属

| 加速项 | Native | PTO |
| --- | --- | --- |
| static_kernel 编译 | 触发，装包 1 个 | **一次都不触发，装包 0 个** |
| SuperKernel | 可用，值 **−54.03 μs**（128K/B16） | **不可用**（`funcHandle is invalid`） |

PTO 的整个 attention 半边是一个自定义算子（`dsv4_hca_forward` → PyPTO kernel），
图里没有可静态化的 aclnn 算子，也没有 SuperKernel 能接受的函数句柄。

⚠ **这是设计使然，不是 PTO 的缺陷。** 用户已明确：static_kernel 与 SuperKernel
都只对 Native 有效，**PTO 用不着**——它靠自己的融合与核内调度，不走这两条。
所以不要为了让 PTO 吃到这两项而改注册方式、拆图或包装算子，也不要把
PTO 侧装包数为 0 当成失败。判据里只有 Native 侧要求实际完成静态编译并装包
（与 CSA 的 `if runtime is None and ...` 一致）。

若把 Native 的 SuperKernel 也开上（上线模板未开，但它确有收益），
128K/B16 的比值会从 1.094 变成 **1.104**（784.31 vs 865.81 首轮口径）。

### 48.3 由此需要重新界定的问题

到 0.80 的差距从旧口径的 76～133 μs 变成 **197～303 μs**。

这个对比本身是公平的：**Native 用它最好的形态（static_kernel + 可选 SuperKernel），
PTO 用它自己的形态（一个融合自定义算子）**，PTO 必须在这个基础上赢。
PTO 用不着那两项，所以问题不是"去把它们接上"，而是
**PTO 自己的这一个融合算子要比 Native 那串带静态 kernel 的 aclnn 算子更快 20%**。

因此需要重算的是：新口径下 PTO 的 915 μs 里，设备侧真正花在 PTO 任务图上的是多少、
主机派发占多少、以及第 39 节那套关键路径分解在新口径下还剩多少可回收空间。
第 36／39／43／44／45 节的绝对数字都要按新口径重做。

## 49. ★★★ 验收口径的七档基线：Native 开 SuperKernel，PTO 比它慢 17.7%（2026-09-29）

第 48 节把两侧都关掉 SuperKernel 当主口径是错的。用户明确：
**sk=0 是供 incore task 细分对比用的，完整的 HCA 差距要开 sk。**
Native 该用它最好的形态；PTO 结构上用不了 SuperKernel（第 47.4 节），照常跑即可。
CSA 的做法一致——其日志原话："分析 incore 时也保持 dynamic=False，只关闭 SuperKernel。"

| 口径 | 用途 |
| --- | --- |
| **Native sk=1 vs PTO** | **完整 HCA 差距 / 验收** |
| Native sk=0 vs PTO | 仅用于 incore task 细分对比（第 48 节那张表） |

### 49.1 验收口径七档（各档各侧 3 轮，取 p50 中位数）

结果目录：Native `results/hca_compiled_20260929/seven_sk1_native/`、
PTO `results/hca_compiled_20260929/seven_sk0/`（PTO 侧 sk 恒为 0，两处 PTO 数据同源）。

| 档位 | Native(sk=1) | PTO | 比值 | 目标 0.80N | 还需降 | 参考 Native(sk=0) |
| --- | --- | --- | --- | --- | --- | --- |
| 128K/B4 | 528.37 | 647.69 | **1.226** | 422.70 | 224.99 | 556.18 |
| 128K/B8 | 599.18 | 727.42 | **1.214** | 479.34 | 248.08 | 662.99 |
| 128K/B16 | 782.34 | 915.34 | **1.170** | 625.87 | 289.47 | 836.60 |
| 128K/B24 | 977.01 | 1046.93 | **1.072** | 781.61 | 265.32 | 1001.22 |
| 8K/B16 | 640.73 | 777.48 | **1.213** | 512.58 | 264.90 | 702.30 |
| 8K/B24 | 760.71 | 907.75 | **1.193** | 608.57 | 299.18 | 797.48 |
| 8K/B32 | 840.78 | 1017.16 | **1.210** | 672.62 | 344.54 | 892.60 |

**128K 平均 1.170，8K 平均 1.206，8:2 加权 1.177。七档全部 > 1.0。**
每档到 0.80 还需降 **225～345 μs**。

### 49.2 SuperKernel 在 Native 上的实测收益

各档（sk=0 减 sk=1）：27.8 / 63.8 / 54.3 / 24.2 / 61.6 / 36.8 / 51.8 μs，**平均 45.8 μs**。
这是 Native 独享的加速，PTO 拿不到（设计使然，见第 47.4 节与记忆
`static-kernel-and-superkernel-are-native-only`）。

### 49.3 与旧口径的对照，以及哪些结论需要重做

| 128K/B16 | 旧口径（不过编译器） | 新口径（验收） |
| --- | --- | --- |
| Native | 671.57 | 782.34（sk=1） |
| PTO | 632.71 | 915.34 |
| 比值 | 0.942 | **1.170** |
| 到 0.80 的差距 | 76～133 μs（七档） | **225～345 μs（七档）** |

⚠ **第 36／39／40／43／44／45 节里所有基于旧口径的绝对数字与预算都作废**
（"缺口 95 μs、可寻址 63 μs、data-wait 63 μs、core-wait 111 μs"等）。
那几节的**结构性结论**（关键路径由哪些任务组成、per-task／per-block 单价的存在与量级、
预取的价值与块数成正比、块数两个方向都不是杠杆）仍然成立，但要在新口径下重新标定。

第 46 节的调度优化待做项清单同样需要按新口径重新估值：门铃并行化在旧口径下预期 −33 μs，
而现在的缺口是 225～345 μs，占比从 1/3 降到 1/10 左右。

## 50. ★★★ 新口径下的拆解：PTO 的设备侧任务图已接近目标，缺口在算子之外（2026-09-29）

给 `dsv4_hca_compiled_case.py` 加了 `--swimlane`（PTO 侧专用）。注意：
**只在 `pypto.torch.init` 里开 `enable_chip_swimlane=4` 不会落盘**，
必须显式 `pypto.torch.begin_dfx()` / `end_dfx()` 包住一次调用，
再用 `dsv4_csa_single_card_bench._export_swimlane` 导出成带真实任务名的泳道
（`kernel_pattern="_jit__decode_hca_tp1_layer_*/kernel_config.py"`）。
这与 `dsv4_hca_performance.capture_swimlane` 的做法一致。

### 50.1 PTO 侧（128K/B16，新口径）

`results/hca_compiled_20260929/swim_pto_h131072_b16/dfx/critical_path_report.md`：

| 量 | 值 |
| --- | --- |
| **设备侧 PTO 任务图 makespan** | **623 μs** |
| 静态 CPM 下限 | 434 μs（69.7%） |
| 观测路径 compute | 485 μs（77.9%） |
| 观测路径 stall | 138 μs（22.1%；data-wait 105、core-wait 32） |
| 端到端 PTO p50（3 轮中位数） | 915.34 μs |
| **任务图之外** | **约 292 μs** |

### 50.2 与旧口径对照：涨的主要不是任务图

| 128K/B16 | 旧口径 | 新口径 | 变化 |
| --- | --- | --- | --- |
| 设备侧任务图 makespan | 538 | 623 | +85 |
| 端到端 PTO | 632.71 | 915.34 | +282.6 |
| **任务图之外** | **95** | **292** | **+197** |

### 50.3 ⚠ 本小节的判断已被第 51 节的 profiler 实测推翻，保留作为过程记录

**PTO 的设备侧任务图 623 μs，已经几乎等于 0.80×Native 的目标 625.87 μs。**

也就是说，在新口径下缺口的主体**不在 PTO 算子内部**，而在它外面：
每次调用的 `context()`（forward context 进出、compact metadata 缓存预置）、
npugraph_ex 对"一个自定义算子"的图重放机制、以及主机派发。
旧 harness 是把调用捕获进自建 NPUGraph 后纯设备重放，这部分开销被排除在外，
所以旧口径看不到它。

⚠ **但还不能据此定论**：Native 侧走同一套计时，也付它自己的那份主机开销，
只是 Native 没有 PyPTO 任务图、无法用同样方式拆。
必须用 torch_npu profiler 把**两侧**的主机/设备分别拆出来对比，
才能判断这 292 μs 里有多少是 PTO 独有的。
两侧的 profiler 采集已提交（`prof_native_h131072_b16` / `prof_pto_h131072_b16`）。

在那份对比出来之前，不要把"缺口在算子之外"当成结论——这是第 41 节同类错误
（当时我用两条不同代码路径相减推断 105 μs 是 compact metadata，实测后是 0）。

## 51. ★★★ profiler 实测：主机开销两侧相同，差距全在设备侧（2026-09-29）

第 50.3 节据"端到端 915 − 任务图 623 = 292 μs"推断缺口主体在 PTO 算子之外。
用 torch_npu profiler 把**两侧**分别拆开后，这个推断**不成立**——
又是一次"拿两个不同来源的数相减"的错误（同第 41 节）。

### 51.1 两侧拆解（128K/B16，profiler 采集，Native sk=1 / PTO sk=0）

| 侧 | kernel 数 | 设备忙时和 | **设备跨度** | 端到端 p50 | **主机/其他** |
| --- | --- | --- | --- | --- | --- |
| Native | 17 | 548.8 | **572.5** | 825.37 | **252.9** |
| PTO | **2** | 1390.3 | **701.2** | 927.40 | **226.2** |

（profiler 会抬高端到端，两侧同受影响；设备跨度取 kernel 记录的首末。）

**主机开销两侧几乎相同（252.9 vs 226.2，PTO 反而低 27 μs）。
差距全在设备侧：572.5 → 701.2，+128.7 μs**，与端到端差 102 μs 同量级。

### 51.2 两侧的设备侧构成完全不同

PTO 侧只有 **2 个 kernel**：

| 耗时 | 核类型 | 名称 |
| --- | --- | --- |
| 701.28 | AI_CPU | `simpler_aicpu_kernel_exec_...`（PyPTO 的 AICPU 调度器，覆盖整个跨度） |
| 689.04 | MIX_AIC | `aicore_kernel_mode_0`（整个融合算子） |

Native 侧 17 个静态编译 kernel，前几名：

| 耗时 | 核类型 | 名称 |
| --- | --- | --- |
| 174.24 | MIX_AIC | `sk_23_..._static_kernel_SparseAttnSharedkv_...` |
| 102.96 | MIX_AIC | `sk_18_..._static_kernel_InplacePartialRotaryMul_...` |
| 63.92 | MIX_AIC | `sk_15_..._static_kernel_RmsNorm_...` |
| 50.34 | MIX_AIC | `sk_13_..._static_kernel_HcPre_...` |
| 44.20 | MIX_AIC | `sk_20_..._static_kernel_Compressor_...` |
| 17.82 | AI_VECTOR_CORE | `triton_rms_kernel` |
| 15.52 | MIX_AIV | `sk_19_..._static_kernel_HcPost_...` |
| 15.50 / 12.14 | AI_CORE | `aclnnMatmulWeightNz_MatMulCommon_MatMulV2` ×2 |

合计 548.8 μs。

### 51.3 ★ 问题的最终形态

**PTO 用一个融合 kernel 做完这半边要 689 μs；Native 用 17 个静态编译 kernel 只要 549 μs。**

按干净口径（无 profiler）折算：主机开销约 240 μs 两侧共有，
则 PTO 设备侧约 675 μs、Native 约 542 μs，而 0.80×Native 对应的 PTO 设备侧预算是
625.87 − 240 ≈ **386 μs**。即 **PTO 的设备侧要从 675 降到 386，−43%**。

这与第 49 节"每档还需降 225～345 μs"一致，并把它定位到了**设备侧融合算子本身**，
不是主机、不是编译路径、也不是 compact metadata。

下一步该做的是逐项对照：Native 那 17 个 kernel 各自做什么、耗时多少，
与 PTO 融合算子内部对应的 incore task 比，找出 PTO 慢在哪几段。
这正是记忆 `perf-target-is-per-incore-task-gap` 说的口径，只是现在有了
Native 侧带静态 kernel 的真实分项数字可以对照。

## 52. ★★★ 差距定位到 O 投影的 cube 效率：PTO 的 proj_a 是 Native 同一矩阵乘的 5.7 倍（2026-09-29）

### 52.1 为什么必须用 sk=0 做细分

sk=1 下 Native 的 24 个算子被 SuperKernel 融合成 **5 个 MIX_AIC 大 kernel（占 79.4%）**，
kernel 名只是被融合那一串的**第一个**（如 `SK:InplacePartialRotaryMul 102.96` 里还含别的算子），
**无法做逐功能归属**。这正是用户说"sk=0 是供 incore task 细分对比用"的原因。
用 `--super-kernel 0` 重采后 Native 展开成 24 个独立 kernel、16 类算子，才能对照。

### 52.2 三个用例的设备/主机拆解（128K/B16，profiler 口径）

| 用例 | kernel 数 | 设备忙时 | 设备跨度 | 端到端 p50 | 主机/其他 |
| --- | --- | --- | --- | --- | --- |
| Native sk=1 | 17 | 548.8 | 572.5 | 825.37 | 252.9 |
| Native sk=0 | 24 | 640.1 | 597.2 | 840.14 | 242.9 |
| PTO | 2 | 1390.3 | **701.2** | 927.40 | **226.2** |

- **主机开销 PTO 最低**（226.2 vs 242.9/252.9），不是差距来源（第 51 节已证）。
- SuperKernel 对 Native 的作用：设备忙时 640.1 → 548.8（−91.3），跨度 597.2 → 572.5（−24.7）。
- ⚠ **不要跨侧比"设备忙时"**：PTO 的两个 kernel（AICPU 调度器 701.28 + AICore 689.04）
  是并发的，忙时和 1390.3 没有意义。**跨度才是可比量。**

### 52.3 逐段对照（Native sk=0 未融合 vs PTO 泳道任务跨度）

| 段 | Native sk=0 | PTO | 差 |
| --- | --- | --- | --- |
| attention | 185.8（`SparseAttnSharedkv`） | 156.2 | **−29.6** ✓ |
| 其余（mHC pre / Q·KV 投影 / RoPE / 压缩器 / KV 写回 / dequant·gather） | 约 366 | 约 274.5 | **−92** ✓ |
| **O 投影 + mHC post** | **约 88** | **193.3** | **+105** ★★★ |

**O 投影这一段的 +105 μs 就等于整个设备侧差距（104 μs）。其余各段 PTO 全赢。**

Native 的 O 投影链（sk=0 下可识别）：

| 算子 | μs |
| --- | --- |
| `aclnnMatmulWeightNz_MatMulCommon_MatMulV2`（wo_a，BF16 NZ） | 19.08 |
| `aclnnDynamicQuantV2_DynamicQuant` | 12.68 |
| `aclnnQuantMatmulWeightNz_QuantBatchMatmulV3`（wo_b，INT8 NZ） | 43.56 |
| `HcPost` | 20.56 |
| 合计 | **95.9**（另一种指派为 81.2，取决于两对同名算子哪个属 Q 哪个属 O） |

PTO 的对应链（泳道跨度 430.7 → 624.0 = 193.3 μs）：

| 任务 | 跨度 | 核·μs | 块 |
| --- | --- | --- | --- |
| `proj_a_mm_0` | 128.2 | **2628.5** | 64（AIC） |
| `quant_0` | 116.8 | 1236.2 | 24（AIV） |
| `_proj_b_mm_nz_kernel__2` | 70.2 | 761.4 | 64（AIC） |
| `hca_oproj_hc_post_0` | 42.5 | 1855.1 | 48（AIV） |

### 52.4 ★ 具体到一个算子：proj_a 的 cube 效率

两者算的是同一件事：`o_packed [96, 8×4096] × wo_a [8, 4096, 1024]` = **3.22 G MAC**。

| | wall | 核·μs |
| --- | --- | --- |
| Native `aclnnMatmulWeightNz` | **19.08** | 约 458（19.08 × 24） |
| PTO `proj_a_mm_0` | 128.2 | **2628.5** |

**即便 PTO 把 24 个 AIC 全占满，2628.5 核·μs 也要 110 μs，是 Native 的 5.7 倍。**
按 a2a3 的 cube 峰值（16×16×16 = 4096 MAC/cycle/核）粗估，Native 的 19.08 μs
已经接近理论值，而 PTO 的 proj_a 只到峰值的约 1/6。

**这是 cube 计算效率问题，不是调度问题**——与第 34～45 节查的派发成本、依赖结构、
预取、块数全都无关。那些方向即便全部做成，量级也只有几十 μs，
而这一项单独就是 105 μs。

### 52.5 与历史记录的关系

第 11／19 节曾否定过 proj_a 的行块 M 分档（v13）与列块 N128→N256（v14/v19），
理由是"cube 本就按 valid_shape 只算有效行"和"NZ 下无带宽可换"。
那些都是在旧口径下、且没有 Native 同一矩阵乘的对照数字时做的判断。
**现在有了 Native 19.08 μs 这个明确靶子，proj_a 的 tiling／数据布局应当重新审视。**
下一步要查的是 PTO proj_a 的 K 维分块、L0/L1 复用与 NZ 权重读取方式，
对照 `aclnnMatmulWeightNz` 的做法（记忆 `compare-pto-against-native-path`）。

## 53. 历史 Native 基线与新口径的对照（2026-09-29）

旧口径（`dsv4_hca_single_layer.py`，不过编译器）历次七档的 Native P50 相当稳定：
128K/B16 一直在 **667～707 μs**、8K/B32 在 **743～771 μs**
（`seven_cann92_v12/v13/v17`、`seven_aligned_v17`、`seven_fulldecode_v17/v22`）。

| 档位 | 旧口径 Native | 新 sk=0 | 新 sk=1 | 新sk1−旧 | 旧口径 PTO | 新 PTO |
| --- | --- | --- | --- | --- | --- | --- |
| 128K/B4 | 405.52 | 556.18 | 528.37 | +122.8 | 409.80 | 647.69 |
| 128K/B8 | 529.23 | 662.99 | 599.18 | +69.9 | 499.73 | 727.42 |
| 128K/B16 | 672.77 | 836.60 | 782.34 | +109.6 | 643.54 | 915.34 |
| 128K/B24 | 868.17 | 1001.22 | 977.01 | +108.8 | 799.79 | 1046.93 |
| 8K/B16 | 538.10 | 702.30 | 640.73 | +102.6 | 528.74 | 777.48 |
| 8K/B24 | 678.54 | 797.48 | 760.71 | +82.2 | 654.05 | 907.75 |
| 8K/B32 | 744.28 | 892.60 | 840.78 | +96.5 | 728.11 | 1017.16 |

两点：

1. **新口径的 Native 反而比旧口径慢 70～123 μs（平均约 99 μs）**，尽管新口径给了它
   static_kernel + SuperKernel。这不是 Native 退步，是计时口径变了：旧 harness 把调用
   捕获进自建 NPUGraph 后纯设备重放、主机开销被排除；新 harness 每次迭代实调编译后的
   module，profiler 量到主机侧 `acl_graph_replay` 326.93、`TorchDynamo Cache Lookup` 79.45、
   `npu_fx_compiler inference` 128.96 μs。
2. **PTO 被抬高得更多**（128K/B16：+271.8 vs Native 的 +109.6），多出的约 162 μs
   正对应设备侧差距（PTO 701.2 vs Native 572.5）。

⚠ 生产用 FULL_DECODE_ONLY 把整步捕获成图重放，**每层的 dynamo lookup 与 fx compiler
调用在重放时不重复付**，所以新口径那 210～250 μs 主机开销两侧都偏高。
但它对两侧大致同量（Native 243～253、PTO 226），差值仍然有效；
而且扣掉主机开销后，PTO/Native 的**设备级**比值是 701.2/572.5 = **1.226**，
比端到端的 1.170 **更差**——去掉主机开销不会帮到 PTO。

## 54. ★★★ 验收口径定稿：只算纯 device 耗时（2026-09-29）

用户明确：**当然只算纯 device 耗时。** 第 48／49 节那些端到端数字里混了每次迭代的
主机派发，不能当验收数字。profiler 拆出来主机侧是
`acl_graph_replay` 326.93 + `TorchDynamo Cache Lookup` 79.45 +
`npu_fx_compiler inference` 128.96 μs，而生产用 FULL_DECODE_ONLY 把整步捕获成图重放，
**这些在重放时不重复付**，属纯粹的测量假象。

### 54.1 口径与实现

- **验收量 = `device_span_us`**：本次重放里第一个 kernel 开始到最后一个 kernel 结束，
  从 torch_npu profiler 的 `kernel_details.csv`（`Start Time(us)` / `Duration(us)`）算出。
  `dsv4_hca_compiled_measure.device_interval()` 负责解析，
  `dsv4_hca_compiled_case.py` 现在**一律采 profiler**并把它写进报告。
- ⚠ **不要跨侧比"设备忙时之和"**：PTO 的 AICPU 调度器 kernel 与 AICore kernel 并发，
  忙时相加没有意义，**跨度才是可比量**。
- 口径写进 `submit_hca_compiled_tiers.sh`：**Native 用 `--super-kernel 1`
  （它最好的形态），PTO 恒为 0（结构上开不了）**；做 incore 细分对比时用
  `SK_NATIVE=0` 覆盖。

### 54.2 ★ 纯设备口径的七档定稿基线（42/42 完成，各档各侧 3 轮取中位数）

结果目录 `results/hca_compiled_20260929/seven_device/`。
Native `--super-kernel 1`，PTO `--super-kernel 0`（结构上开不了）。

| 档位 | Native 设备 | PTO 设备 | 比值 | 0.80N | 还需降 | 端到端比值（参考） |
| --- | --- | --- | --- | --- | --- | --- |
| 128K/B4 | 297.2 | 411.8 | **1.385** | 237.8 | 173.9 | 1.163 |
| 128K/B8 | 393.8 | 538.8 | **1.368** | 315.0 | 223.8 | 1.186 |
| 128K/B16 | 550.5 | 687.8 | **1.249** | 440.4 | 247.3 | 1.163 |
| 128K/B24 | 754.8 | 829.0 | **1.098** | 603.8 | 225.2 | 1.122 |
| 8K/B16 | 400.5 | 551.2 | **1.376** | 320.4 | 230.8 | 1.254 |
| 8K/B24 | 538.8 | 691.0 | **1.283** | 431.0 | 260.0 | 1.186 |
| 8K/B32 | 620.5 | 759.5 | **1.224** | 496.4 | 263.1 | 1.242 |

**128K 平均 1.275，8K 平均 1.294，8:2 加权 1.279。
七档合计还需降 1624.2 μs，单档均 232.0 μs。**

轮间极差：Native 3.8～27.0、PTO 8.2～49.0 μs，数据可信。

### 54.4 比值随 batch 单调下降 → PTO 有一块不随 batch 缩小的固定成本

128K：B4 1.385 → B8 1.368 → B16 1.249 → B24 1.098。
8K：B16 1.376 → B24 1.283 → B32 1.224。

Native 的算子链随 token 数线性缩小，而 PTO 有一块固定成本摊不开。
这与第 45 节量到的派发单价一致：**每级转换 ≈ 2.3 μs + 0.27 μs × 块数**，
而任务数（48）与块数（664）由算子结构决定、不随 token 数变化。
按 128K/B4 的 Native 设备侧只有 297 μs 算，这块固定成本的占比相当可观。

### 54.3 纯设备口径下比值明显更差

因为两侧共有的约 250 μs 主机开销在端到端里把比值往 1 拉。小 batch 尤其明显：
128K/B4 的 Native 设备侧只有 297 μs，**主机开销比设备工作本身还多**，
端到端比值 1.163 而纯设备是 1.385。

**这也说明：想靠削减主机开销改善验收数字是无效的——验收本来就不看它。**
第 50／51 节围绕主机开销的讨论到此收束：它既不是差距来源，也不在验收口径内。

## 55. 核内第一轮：proj_a 的 K 维分块——L0B 的约束摸清了，但加深流水无效（2026-09-29）

目标（用户 2026-09-29 定）：**七档达到 Native 的 0.80 倍，核内优化与核间流水轮流做。**
本节是核内的第一轮，对象是第 52 节定位到的 `proj_a_mm_0`。

基线（128K/B16，PTO 侧纯设备跨度，3 轮取中位）：**673.0 μs**
（671.5 / 673.0 / 689.2，极差 17.8）。目标 0.80×550.5 = **440.4 μs**。

### 55.1 出发假设与三个候选

`proj_a` 每个 K 步的 L0 占用：
L0A = `PROJ_A_ROW_TILE(128) × A_K_TILE × 2B`，L0B = `A_K_TILE × A_COL_TILE × 2B`。
base 用 K=256、N=128 → 两者都是 **64 KiB，正好占满物理容量**，
而循环写的是 `pl.pipeline(..., stage=2)`。假设是双缓冲放不下、搬运与计算串行。
（同文件精度版第 88 行有同类先例："L0C 也从满格的 128KiB 降到 64KiB 留出双缓冲"。）

| 候选 | K | N | L0B 每片 | 结果 |
| --- | --- | --- | --- | --- |
| kt128 | 128 | 128 | 32 KiB | **编译失败**：`Right buffer usage (131072) exceeds platform limit (65536)` |
| kt64 | 64 | 128 | 16 KiB | 编译通过，**710.8 μs，比 base 差 +37.8（+5.6%）** |
| kt128n64 | 128 | 64 | 期望 16 KiB | **编译失败，报的仍是 131072** |

### 55.2 由此摸清的三条 L0B 约束（都是实测，不是推测）

1. **`stage=2` 实际要 4 片**：kt128 每片 32 KiB 报 131072 = 4×32 KiB；
   kt64 每片 16 KiB → 4×16 = 64 KiB 正好合规、编译通过。
2. **base 的 K=256/N=128 每片 64 KiB 时，分配器只给 1 片**（否则早就超限），
   即 **base 事实上没有双缓冲**——原假设的前提成立。
3. **NZ 布局下 N 的切片粒度不小于 128**：kt128n64（N=64）报的数字与 kt128（N=128）
   **完全相同**（131072），说明 N=64 没让片子变小。**N 不能再往下调。**

### 55.3 ⚠ 但假设的结论被推翻：加深流水反而更慢

kt64 真正拿到了 4 片缓冲，却比只有 1 片的 base **慢 37.8 μs**。
说明 proj_a 不是"搬运与计算串行"受限，而是 K 切细后的指令与同步开销超过了重叠收益
（K=64 时 K 步数从 16 涨到 64）。

**"给 proj_a 加深 L0 流水"这条路到此关闭。** 与第 0.2 节里 v13（行块）、v14/v19（列块）
一起，proj_a 的 tiling 三个维度（M/N/K）都各有一次失败记录；
但注意 v14/v19 是旧口径下测的，且当时没有 Native 的对照数字。

### 55.4 ⚠ 同时修正第 52.4 节对 Native 侧算子的归属

第 52.4 节说 Native 的 wo_a 是 `aclnnMatmulWeightNz`（19.08 μs），据此算出 5.7 倍差距。
按 MAC 量重新核对，这个归属**不对**：

| Native 侧 kernel（sk=0） | μs | 按 MAC 量推断 |
| --- | --- | --- |
| `aclnnTransposeBatchMatMulWeightNz_TransposeBatchMatMul` | 51.54 | **wo_a**（8 组批量，3.22 G MAC，BF16 NZ） |
| `aclnnQuantMatmulWeightNz_QuantBatchMatmulV3` ×2 | 43.56 / 31.16 | wq_b 与 wo_b（各 3.22 G MAC，INT8 NZ） |
| `aclnnMatmulWeightNz_MatMulCommon_MatMulV2` ×2 | 19.08 / 16.80 | wq_a（0.40 G）与 wkv（0.20 G），都很小 |

所以 **PTO 的 proj_a 对标的是 51.54 μs 的批量矩阵乘，不是 19.08 μs**：

| | wall | 核·μs（按 24 核折算） | 每核 M MAC/μs |
| --- | --- | --- | --- |
| Native `TransposeBatchMatMul`（wo_a） | **51.54** | 约 1237 | 2.6 |
| PTO `proj_a_mm_0` | 128.2 | **2628.5** | 1.22 |

**差距是 2.1 倍（wall 2.5 倍、约 +77 μs），不是 5.7 倍。** 两侧都远离 cube 峰值
（约 7 M MAC/μs/核），Native 只是效率高 1.6 倍。
O 投影整段：Native 51.54 + 12.68（DynamicQuant）+ 43.56（wo_b）+ 20.56（HcPost）
= **128.3 μs**，PTO 193.3 μs，**+65 μs**。

⚠ 这个归属是**按 MAC 量推断**的，不是 profiler 给出的标注映射
（`operator_details.csv` 只有 8 条主机侧条目，没有算子到 kernel 的对应）。
要坐实需要读 `ascend_pytorch_profiler_0.db` 的关联表。

### 55.5 下一个核内候选：提高算术强度而不是缓冲深度

既然 N 不能变小、K 变细更慢，改走**提高每次 A 片复用度**：
N 从 128 提到 **256**，A 片被 256 列复用而不是 128 列，**A 侧 MTE2 流量减半**；
配 K=128 使 L0B 每片 128×256×2 = 64 KiB 刚好合规（单片，与 base 同缓冲深度）。
候选 `source_k128n256` 已提交三轮。

## 56. 核间第一轮：补预取无效（纯设备口径复现 v34a）；并把 +122 μs 拆到两个大头（2026-09-29）

### 56.1 候选 early7：给压缩器/KV 写回区的 7 处补 `allow_early_resolve`

对象是新口径关键路径上 stall 最大的那一段。时间轴（PTO 侧新口径泳道）：

| 任务 | 窗口 | 核·μs |
| --- | --- | --- |
| `hca_softmax_pool` | 148.2→156.5 | 100.7 |
| `hca_raw_cache_write` | 154.4→173.4 | 139.3 |
| （AIV 空闲 7.9 μs） | 173.4→181.3 | — |
| `hca_norm_rope_write` | 181.3→185.0 | 46.8 |
| `hca_inverse_rope_sign` | 185.2→189.6 | 51.2 |
| `hca_state_commit` | 187.3→195.0 | 114.5 |

`156.5→195.0` 共 **38.5 μs 墙钟只做了 7.3 μs 的打包 AIV 工作**，且这五个任务
全都没有 `allow_early_resolve`。按第 45 节的单价 `2.3 + 0.27×16 = 6.6 μs` × 5 ≈ 33 μs，
与浪费量吻合，所以补上标记应当见效。

| 变体 | 设备跨度中位 | 各轮 | 相对 base |
| --- | --- | --- | --- |
| base | **673.0** | 671.5 / 673.0 / 689.2 | — |
| early7 | 683.0 | 667.2 / 683.0 / 697.0 | **+10.0（+1.5%），区间大幅重叠 → 无效** |

**结论：无效。** 这在纯设备口径下复现了 v34a 的结果（当时 −1.07 μs，曾被怀疑是
端到端口径把差异稀释了；现在证明不是口径问题）。
**那 31 μs 的 stall 不是派发延迟**，`2.3 + 0.27×块数` 的单价模型不能外推到这一段。
第 45.2 节"补标记的收益与块数成正比"这一说法要收窄为：**只在 v21 那三个大块任务上验证过，
不能推广。**

### 56.2 把 +122.5 μs 的差距拆到任务

PTO 设备 673.0 vs Native(sk=1) 550.5，差 +122.5 μs。按关键路径拆：

| 段 | PTO | Native sk=0 | 差 |
| --- | --- | --- | --- |
| attention | 155.7 | 185.8（`SparseAttnSharedkv`） | **PTO 赢 30.1** |
| **`proj_a_mm_0`（路径上出现 3 次）** | **113.5** | 51.5（`TransposeBatchMatMul`） | **输 62.0** |
| **`hca_cmp_work_gather_0`** | **57.1** | 0（融进 SparseAttn，无独立算子） | **输 57.1** |
| quant + proj_b + oproj_post | 约 90 | 76.8 | 输 13.2 |

**两个大头：proj_a 的 +62 与 cmp_work_gather 的 +57，合计 119 μs ≈ 全部差距。**

### 56.3 ★ proj_a 是 cube 吞吐受限，不是搬运受限

`proj_a` 打包核时 = 2628.5 / 24 = **109.5 μs**，而关键路径上它占 **113.5 μs**
（#9/#10/#11 三次，中间 stall 为 0）。两者几乎相等，说明 **占核率约 100%**。

推论：
- **流水/重叠改不动它**（8 个组任务在 24 个 AIC 上是 2.67 波，已经背靠背）；
- **提高 A 复用、减少 MTE2 流量也改不动它**（不是搬运受限）；
- 只能**减少 MAC 数**或**提高每 MAC 的 cube 效率**。

### 56.4 下一个核内候选 m96：验证"补位行到底算不算"

`PROJ_A_ROW_TILE = 128`，而 96 token 只有 96 行有效——**25% 的 cube 工作在补位行上**。
第 11 节 v13 否掉 M 分档的理由是"cube 本就按 `valid_shape` 只算有效行"。
这个说法可以一测定音：把 `PROJ_A_ROW_TILE` 改成 96（= 6×16，满足 cube 行粒度，
`proj_a_rows = ceil(96/96) = 1`，块数不变）。

- 若 v13 的说法成立 → 本候选**毫无变化**；
- 若不成立 → proj_a 的 cube 工作降约 25%，即 109.5 × 0.25 ≈ **−27 μs**。

`source_m96` 已提交三轮。

## 57. ★ proj_a 的 tiling 空间已被实测封死；CSA 式 gather 融合经算不可行（2026-09-29）

### 57.1 m96 与 k128n256s1：两个负结果各自封掉一个维度

| 候选 | 改动 | 设备跨度中位 | 相对 base（673.0） |
| --- | --- | --- | --- |
| m96 | `PROJ_A_ROW_TILE` 128→96 | 677.8 | +4.8（噪声内） |
| k128n256s1 | K=128、N=256、显式 `stage=1` | 701.8 | **+28.8（+4.3%）更差** |

**m96 确认了 v13 的说法**：cube 确实只按 `valid_shape` 算有效行，
96 token 下 M=128 没有 25% 的补位浪费。这个负结果有价值——它排除了"补位浪费"这个解释。

**k128n256s1 变差的原因是 L0C**：N=256 时累加器 = M(128)×N(256)×4B = **131072 = 128 KiB，
正好占满整个 L0C**，没有双缓冲余量；base 的 N=128 只占 64 KiB（一半）。
这与精度版 `decode_o_proj.py:88` 那句注释说的是同一件事
（"L0C 也从满格的 128KiB 降到 64KiB 留出双缓冲"）。

### 57.2 proj_a 三个维度全部被钉住（都是实测边界）

| 维度 | 下界 | 上界 | 结论 |
| --- | --- | --- | --- |
| N | **≥128**：NZ 切片粒度，kt128n64（N=64）与 kt128 报同一个 131072 | **≤128**：L0C 双缓冲，k128n256s1 变差 | **只能是 128** |
| K | — | **≤256**：L0B 每片 ≤64 KiB；K=128 分配器要 2～4 片必超限，强制 stage=1 更差；K=64 得 4 片但慢 37.8 | **只能是 256** |
| M | — | — | 96 与 128 无差别（cube 只算有效行） |

**`proj_a_mm` 现有形式下已无可调空间**，五个候选（kt64 / kt128 / kt128n64 / k128n256 /
k128n256s1 / m96）全部否定。要再动 proj_a 必须改算法结构
（Mat/L1 暂存 A 配 N 内循环，或按 K 切块跨块归约），都是较大改动且各有代价：
前者会把并行块数从 64 降到 8（只用 8 个核），后者要引入归约、改数值。

⚠ 也要修正第 56 节的带宽推断：那里据"每块 2 MiB / 41.1 μs = 1.17 TB/s"
认为 proj_a 是 HBM 带宽受限、A 重读 8 次是主因，并预测 N 加倍能省约 32 μs。
**k128n256s1 把 A 重读从 8 次降到 4 次，结果更差 28.8 μs**，所以那个推断即便方向对，
也被 L0C 的约束压过去了——**不能把它当作已确认的机制。**

### 57.3 CSA 式 gather 融合：经计算不可行，不必实现

`hca_cmp_work_gather` 在路径上占 57.1 μs，而 Native 把 gather 融进了 `SparseAttnSharedkv`
内核、没有独立算子。自然的想法是照 CSA 的 `decode_sparse_attn_csa.py`（L199-330）
把分页 gather 搬进 attention 的 AIV lane，用 FFTS 逐块交接。

**算过之后不可行**：attention 的 tick 循环是
`for token in pl.range(worker, t_dim, NUM_QK_CORES)`，每个 worker 的 4 个 token
**来自不同请求**（步长 24），同一请求的 6 个 token 不共享 KV 载入。
所以暂存张量 `cmp_work_kv` 的存在正是为了**避免分页 gather 被做 6 遍**
（每请求 6 个 token）。融进去等于把 771.8 核·μs 变成约 6 倍。

这也解释了为什么 CSA 能融而 HCA 不能：**CSA 的 topk gather 本来就是逐 token 的**，
没有跨 token 复用可言；HCA 的压缩历史是整请求共享的，先暂存再复用才是对的。

**结论：不要实现这个融合。** 记入第 0.2 节。

## 58. ★★★ 根因：两条 lane 都是 GM 带宽受限，算子调优已到局部最优（2026-09-29）

### 58.1 六轮候选全否，模式一致

| 类别 | 候选 | 结果 |
| --- | --- | --- |
| 增加块数／并行度 | v35a、qt8、qt16、kt64 | 全部更差（+17～+37 μs） |
| 减少块数 | v29 | 更差 |
| 补预取 `allow_early_resolve` | v34a、early7 | 中性 |
| proj_a tiling（M/N/K） | kt128、kt128n64、k128n256、k128n256s1、m96 | 编译失败或更差 |
| 改依赖／拆栅栏 | v30、v37 | 中性 |

**算子已处在局部最优**：往任何方向调块数、tiling、预取、依赖都不会赢。
这不是候选选得差，而是这一版算子被这个团队调过很多轮了。

### 58.2 用带宽把 attention 的耗时算一遍，模型闭合

`hca_unified_attention` 是最大单项（AIV 7323 + AIC 3554.7 = 10878 核·μs，占全部核时约 44%）。

**AIC 侧：3554.7 / 24 = 148 μs**

| 项 | 量 | 估时 |
| --- | --- | --- |
| KV 载入（`kv_l1` 从 `cmp_work_kv` gather，144 KiB/tick × 36 tick × 24 块） | **124 MiB** | 约 95 μs @1.3 TB/s |
| cube（QK+PV 每 tick 8.4 M MAC × 36 = 302 M MAC/块，@6.55 M MAC/μs） | — | 约 46 μs |
| **合计** | | **141，实测 148** ✓ |

**AIV 侧：7323 / 48 = 152.6 μs**

| 项 | 量 | 估时 |
| --- | --- | --- |
| `scores` 读 16 KiB + `probs` 写 8 KiB + **`values` 读 64 KiB**，每 tick 88 KiB × 36 × 48 | **152 MiB** | 约 117 μs |
| exp / 在线重标定 | — | 余量 |
| **合计** | | **约 152.6** ✓ |

⚠ **这个带宽模型未经验证，不要当结论。** 两条 lane 的估时都用了 1.3 TB/s（HBM 量级），
但交接缓冲 `scores`/`probs`/`values` 合计只有约 1.5 MiB
（`transfer_rows = NUM_QK_CORES × QK_TRANSFER_SLOTS × H`），**完全装得进 192 MiB 的 L2**，
往返很可能走 L2 而非 HBM——那样时间是约 30 μs 而不是 117 μs，"模型闭合"就是巧合。
同理 AIC 侧 `cmp_work_kv` 只有约 18.9 MiB，也在 L2 内。

要验证需要 pipe 级数据（MTE2 vs cube 各占多少），level-4 chip swimlane 不提供，
而 PTO 整个半边只有一个 kernel，torch_npu profiler 也拆不开。
**在拿到 pipe 级证据之前，"attention 是 GM 带宽受限"只是假设。**
（第 41、50 节各有一次"用两个不同来源的数相减就下结论"的教训，这里不重复。）

唯一**已验证**的相关事实：`pl.create_tensor(shape, dtype, layout, manual_dep, init_value)`
**没有内存空间参数**，所以暂存与跨核交接张量按设计驻留 GM，算子侧无法指定 L2/UB。
（tile 级的 `pl.load(..., target_memory=...)` 只影响搬进来之后放哪，不改变背后的张量位置。）

### 58.3 两个结构性根因

1. **AIC 的 KV 被重读 6 次**：`for token in pl.range(worker, t_dim, NUM_QK_CORES)`
   让每个 worker 的 token **跨请求**（步长 24，token t 属请求 t//6），
   同一请求的 6 个 token 由 6 个不同 worker 各自 gather 一遍压缩历史。
   唯一 KV 只有约 21 MiB，实际读 124 MiB。
2. **AIC↔AIV 的交接走 GM**：`scores`／`probs`／`values` 都是 `pl.create_tensor`（GM 驻留），
   其中 `values` 的 FP32 占 AIV 流量的 73%（64 KiB / 88 KiB per tick）。
   Native 的 `SparseAttnSharedkv` 把这些留在 L1/UB 内，**一次 GM 往返都不做**——
   这正是它 185.8 μs 能做完整个 attention 的原因。

### 58.4 ⚠ "把一个请求的 6 个 token 批在一起"经计算不值得做

想法：worker 按请求分配，一次 QK/PV 处理该请求的 6 个 token
（M 从 64 提到 384，KV 每请求-tile 只载一次）。前提成立——
`cmp_rows = min(HCA_MAX, min((position+1)//RATIO, length//RATIO))`，
RATIO=128 而请求内 position 只差 ≤5，**压缩部分对 6 个 token 几乎完全相同**。

但算完收益很小：

| | 现在 | 批 6 token 后 |
| --- | --- | --- |
| AIC | 148 μs（95 搬运 + 46 cube） | 约 62 μs（16 搬运 + 46 cube） |
| AIV | 152.6 μs | **152.6 μs（不变——交接字节总量不变，只是 tick 数少 6 倍、每 tick 大 6 倍）** |
| attention 跨度 = max | 156 | **约 153** |

**净收益约 3 μs，而这是对 attention 内核的大改写。不做。**

要让 AIV 也降下来，只能减小交接缓冲：把 `values` 从 FP32 改 BF16 可让 AIV 流量减半
（约 −44 μs），但**会改变在线 softmax 的累加精度**，与整机 token 逐 bit 验收冲突。

### 58.5 对目标可达性的判断

验收要求 PTO ≤ 0.80×Native，当前纯设备口径 8:2 加权 **1.279**，即需要提速 **1.60 倍**。

已确认的事实：
- 算子的调优空间（块数、tiling、预取、依赖）**已经用尽**，六轮全否；
- 主要耗时项（attention 两条 lane、proj_a）**都已到 GM 带宽上限**；
- PTO 与 Native 的差距来自**结构**：PyPTO 的跨核交接与暂存张量驻留 GM，
  而 Native 的 SuperKernel/静态 kernel 把同样的中间结果留在 L1/UB。

**结论：1.60 倍不可能靠算子内调优达成**——这一条由六轮全否的实测支撑，与带宽假设是否成立无关。
可能的路径只有两类，都超出"调算子"的范围：

1. **框架层**：让 PyPTO 的跨核交接缓冲（`scores`/`probs`/`values`）与暂存张量
   （`cmp_work_kv`）驻留在 L2 或 UB 而非 GM。这是 PyPTO/Simpler 的能力问题，
   用户已授权改 simpler（见 `branch-before-touching-simpler`），但这属于运行时/编译器特性，
   不是参数调整。
2. **算法层**：减少中间结果的数据量（例如交接用 BF16），代价是改变数值，
   与整机 token 逐 bit 验收冲突，需要用户裁决。

---

## 59. ★★★★ pipe 级取证推翻 §56/§58：PTO 的核内计算已与 Native 持平，差距全在占用率（2026-09-29）

§58 把根因判为"两条 lane 都是 GM 带宽受限"，并自己标注了"在拿到 pipe 级证据之前这只是假设"。
证据拿到了，**假设是错的**，而且 §56.3「proj_a 是 cube 吞吐受限」也一并被推翻。

### 59.1 取证手段：torch_npu profiler 的 AiCMetrics

此前我在日志里写过"PTO 整个半边只有一个 kernel，torch_npu profiler 也拆不开"——
这句话对"按算子拆分"是对的，但对"按 pipe 拆分"是错的。
`torch_npu.profiler.AiCMetrics` 提供 `PipeUtilization`、`L2Cache`、`MemoryAccess`
等选项，开了之后 `kernel_details.csv` 会对**每个 kernel**多出
`aic_mac_time/ratio`、`aic_mte1/mte2_time/ratio`、`aic_fixpipe_*`、`aic_scalar_*`、
`aiv_vec_*`、`aiv_scalar_*`、`aiv_mte2/mte3_*`、`cube_utilization(%)` 等列。
PTO 只有一个 `aicore_kernel_mode_0`，但它的 pipe 占比同样被采到，
**足以判定整层受哪条 pipe 限制**。

已把 `--aic-metrics` 接到 `dsv4_hca_compiled_case.py`，
`dsv4_hca_compiled_measure.py` 的 `device_interval` 会按耗时加权把这些列聚合进
`report.json` 的 `device_profile.top[].metrics`。

口径：128K/B16，两侧都 `--super-kernel 0`（要逐算子对照，必须关 SuperKernel），
`--iters 20 --warmup 5`。PTO span 699.25 μs（L2 那次 676.5），
Native sk=0 span 599.5 μs（L2 那次 590.5）——与既有基线闭合
（Native sk=1 550.5 + SuperKernel 收益 54.3 = 604.8）。

### 59.2 每核 pipe 时间：只有 scalar 一条超出，其余全部持平或更优

Native 侧按 `n`（调用次数）加权求和，得到"整层每核累计"，与 PTO 的单 kernel 直接可比。

| 每核 pipe 时间 (μs) | PTO | Native sk=0 | 判读 |
|---|---:|---:|---|
| AIC aicore_time | 683.5 | 374.2 | PTO 的核自始至终不退出 |
| AIC mac | **97.6** | **101.6** | 真实 cube 运算量持平 |
| AIC mte1 | 111.9 | 114.9 | 持平 |
| AIC mte2 | **252.6** | **288.2** | PTO 反而更少 |
| AIC fixpipe | 73.8 | 184.4 | PTO 少一半以上 |
| AIC scalar | **363.8** | **156.9** | ← 唯一超出项 |
| AIV aiv_time | 685.3 | 462.1 | 同上 |
| AIV vec | 114.8 | 75.7 | +39.1 |
| AIV mte2 | **77.3** | **80.7** | PTO 更少 |
| AIV mte3 | 34.7 | 40.8 | PTO 更少 |
| AIV scalar | **387.3** | **76.8** | ← 唯一超出项 |
| cube_utilization | **95.07%** | 61.9～91.0% | PTO 比 Native 任何算子都高 |

换成核·μs 的总 MAC 量（按各算子实际 Block Num 加权）：

- PTO：97.567 × 24 核 = **2341.6 核·μs**
- Native：SparseAttnSharedkv 47.533×24 = 1140.8；QuantMatmulWeightNz 12.135×19.6×2 = 475.7；
  TransposeBatchMatMulWeightNz 18.689×24 = 448.5；MatmulWeightNz 3.003×22×2 = 132.1；
  Compressor 2.759×24 = 66.2；HcPre 2.387×24 = 57.3 → 合计 **2320.6 核·μs**

**PTO 与 Native 的实际 cube 运算量相差 0.9%。** 这直接否掉 §56.3 与
`hca-optimize-alternate-incore-and-pipeline` 记忆里写的
"proj_a_mm_0 用 2628.5 核·μs 做 3.22 G MAC，Native 只用 458 核·μs，差 5.7 倍"——
那是拿「task wall × 核数」去比「MAC pipe 时间」，两个量根本不同类。
proj_a 的 2333.7 核·μs 里真正的 MAC 只占一小部分，其余是等待。

### 59.3 带宽假设的直接证伪

`--aic-metrics L2Cache` 给出 PTO 的 `aicore_kernel_mode_0`：

| 通道 | hit | miss | 命中率 |
|---|---:|---:|---:|
| aic r0 read | 1248052 | 432019 | 74.3% |
| aiv r0 read | 858659 | 62690 | 93.2% |
| aic write | 254337 | 80354 | 76.0% |
| aiv write | 48710 | 92134 | 34.6% |

AIC 读 miss 432019 次，按 128 B 行算约 55 MB，在 670 μs 里是 **约 82 GB/s**，
而 a2a3 的 HBM 在 TB/s 量级。**PTO 离 HBM 带宽饱和差两个数量级。**
加上上表里 PTO 的 mte2 时间本来就比 Native 少，
§58.2 那套"用 1.3 TB/s 把 attention 耗时算一遍、模型闭合"的推导是巧合，不成立。
（§58 自己已标注"transfer 缓冲约 1.5 MiB、cmp_work_kv 约 18.9 MiB 都能装进 192 MiB L2，
所以 1.3 TB/s 这个前提可能是错的"——现在确认就是错的。）

### 59.4 scalar 超出的来源：核在自旋等依赖，被计进 scalar pipe

`aic_total_cycles` 30345844 ÷ 24 核 = 1.264 M cycles，按 ~1.85 GHz 折算约 683 μs，
**等于 kernel 的整个生命期**——说明 PTO 的核从 kernel 进入到退出一直在计时忙，
从不休眠。等依赖的自旋轮询走的是 scalar pipe。

拿 128K/B16 的 level-4 泳道核对（按 CoreId 统计每核真实执行时间）：

| | 真实忙（核·μs） | 每核忙 (μs) | kernel 生命期 | 每核空转 |
|---|---:|---:|---:|---:|
| AIC（24 核） | 10780.4 | 449.2 | 683.5 | 234.3 (34.3%) |
| AIV（48 核） | 16714.4 | 348.2 | 685.3 | 337.1 (49.2%) |

把空转当作纯自旋扣掉后的"真实地址计算 scalar"：

- AIC：363.8 − 234.3 = 129.5 μs，占忙时 449.2 的 **28.8%**（Native 156.9/374.2 = 41.9%）
- AIV：387.3 − 337.1 = 50.2 μs，占忙时 348.2 的 **14.4%**（Native 76.8/462.1 = 16.6%）

**即 PTO 的真实地址计算开销比例还低于 Native。** 结论只剩一条：
**差距不在核内，在占用率——核有三分之一到一半的时间在等。**

### 59.5 结论与方向调整

1. **核内这条轴已经到位**：mac / mte1 / mte2 / fixpipe / cube_utilization 全面持平或优于
   Native。继续在 tiling、K 分块、NZ 读法上找收益，预期回报接近零
   （这也解释了 §51～§57 连续否掉八个候选的模式）。
2. **全部剩余空间在核间流水**。§58.5 对"目标可达性"的悲观判断建立在带宽假设上，
   该判断作废。
3. 后续核内工作只保留一项有意义的：**`aiv_vec` 114.8 vs Native 75.7（+39.1 μs/核）**。
   这是唯一真实超出的计算型 pipe，来源疑为几处用 `arange`/`cast`/`col_expand_add`/
   `row_expand_add` 现场构造 gather 索引的写法（例如 `hc_pre_fused.py` 的
   `mix_x_rms_norm`，注释说是为了绕开"TEXTRACT 从 UB 的非 32B 对齐列读取"）。
   量级 39 μs/核，不是大头，排在流水之后。

---

## 60. ★★★ 占用率拆解：前半段 195 μs 富余、尾段 55 μs，`qproj_matmul` 是最大单项（2026-09-29）

§59 把方向定到占用率之后，用 128K/B16 的 level-4 泳道按 CoreId 逐核统计
（脚本思路：从 `event-hint` 里解析 `CoreId:<n>`，AIC = 0–23，AIV = 24–71，
按 family 统计块数、真实核集合、核·μs，再按区段算并发核数）。

### 60.1 三个区段各自的下限

| 区段 | 观测 wall | AIC 核·μs → ÷24 | AIV 核·μs → ÷48 | 区段下限 | 富余 |
|---|---:|---:|---:|---:|---:|
| 前半段 0→354.9 | 354.9 | 3830.1 → 159.6 | 6179.0 → 128.7 | 159.6 | **195.3** |
| attention 354.9→521.8 | 166.9 | 3768.4 → 157.0 | 7751.4 → 161.5 | 161.5 | 5.4 |
| 尾段 517.3→704.5 | 187.2 | 3181.9 → 132.6 | 2784.0 → 58.0 | 132.6 | 54.6 |

**attention 已经跑到 97% 的核效率**，且 PTO 的 166.9 μs 本来就快于 Native 的
`SparseAttnSharedkv` 187.7 μs。这条 lane 没有空间，也不该再碰。

尾段并发核数实测：均值 32.5 / 72，p50 34，空转 7429 核·μs（容量 13476）——
**不是容量问题**。原因是 8 个 proj_a（各 8 块）+ 8 个 proj_b（各 8 块）共 64 块
同时抢 24 个 AIC 核，每个 task 各拿 7–8 核，排队极不均匀：
同一批 proj_a 的 8 个 task，核·μs 分别是 352.1/365.7/250.5/214.0/239.3/281.6/283.7/346.8，
而 wall 是 47.2/48.9/**101.2**/30.9/31.9/67.1/43.8/68.6——250.5 核·μs 的那个 task
被拖到 101.2 μs。

### 60.2 ⚠ 更正：proj_a / proj_b / quant 不是"64 块 3 波"

第一版分析我按 `name` 聚合，得出"proj_a_mm 64 块落在 24 核 = 3 波"。
**错。** 按 `taskId` 拆开后是 **8 个独立 task 各 8 块**（O_GROUPS = 8，
`proj_a_rows × (O_LORA // A_COL_TILE)` = 1 × 1024/128 = 8）。
quant 是 8 个 task 各 3 块，proj_b 是 8 个 task 各 8 块。
以后做这类统计必须按 taskId 拆，`name` 会把 `pl.parallel` 下的同名 task 合成一个。

### 60.3 ★ `qproj_matmul`：24 块硬编码遇上 20 个空闲核，wall 翻倍

`deepseek_v4_flash_dspark/q_projection.py:30 QPROJ_WORKERS = 24`，
`pl.spmd(QPROJ_WORKERS)` 恒为 24 块；`QPROJ_N_BLOCKS = H*HEAD_DIM/QPROJ_MM_N_TILE
= 32768/256 = 128` 个 N 块按 24 取模余 8，所以 8 个 worker 做 6 轮、16 个做 5 轮。

真正的问题不是这 6/5 的不均，而是**派发时刻 AIC 核不够**：
`hca_kv_score_proj`（16 块，AIC，139.2→185.6）还在跑，qproj 在 168.2 派发时
只抢到 **20 个** AIC 核，于是 24 块里有 4 块排到第二轮，各自再跑一个完整的
~55 μs，wall 从 59.6 变成 **112.2**（1272.9 核·μs ÷ 20 核的下限只有 63.6）。

七档里这个"核不够"的情形只在 **128K/B16（20 核）与 128K/B24（21 核）**出现，
恰好是 8:2 加权里最重的两档。其余五档都拿到 24 核。

### 60.4 实测：w20 / w32 在 B16 拿到 −28.75 μs，B24 无效

变体机制：把 `vllm_ascend/ops/pypto` 整树拷到 `tests/pypto_test/variants_qproj_20260929/<v>/`，
只改 `QPROJ_WORKERS`，用 `--operator-source` 指过去（@pl.jit 编译时重读源文件，
必须这样隔离，见 `pto-jit-rereads-source-pin-variant`）。

| | base(24) | w20 | w32 | w48 |
|---|---:|---:|---:|---:|
| 128K/B16 span_us | 688.75 | **660.00** | **663.00** | 707.50 |
| 128K/B24 span_us | 838.25 | 835.50 | 833.00 | 838.75 |

四个变体的逐 bit 对照全部 PASS（`eager_comparison` 全 PASS），
这是纯块数变更，不动数值。

B16 的 −28.75 μs 超过噪声（进程间 p50 stdev 6～13 μs，span 单次重放同量级）；
B24 的 −2.75/−5.25 在噪声内，说明 **B24 的 qproj 不在关键路径上**，
它那 163.1→91.1 的富余被别的东西盖住了。

按每轮成本算，小 T 档反而不该降 worker：B4 的 qproj 是 929 核·μs、
每轮 7.26 μs，w20 要 7 轮 = 50.8 μs，比现在的 wall 47.9 还差。
所以这个常量是否该改、改成多少，要等七档加权结果。已提交七档 × {base, w20, w32}。

### 60.5 ✗ 负结果：链上三个 12 块的 AIV 任务都被"8 行 = 32 B 对齐"钉死

前半段链上有三个任务只用了 48 个 AIV 核里的 12 个，本来是最像"免费提速"的地方：

| 任务 | 块数来源 | 核·μs | wall | 若铺到 48 核 |
|---|---|---:|---:|---:|
| `hca_hc_widen_rms` | `ceil(tokens/WIDEN_ROWS=8)` | 336.3 | 28.9 | 7.0 |
| `mix_x_rms_norm` | `t_pad // T_TILE=8` | 469.0 | 40.1 | 9.8 |
| `qr_rms_norm_quant` | `ceil(tile_rows/T_TILE=8)`，再被 `QR_NORM_WORKERS=16` 截 | 383.7 | 32.6 | 8.0 |

**三个都不能改**，各有硬约束：

1. `hca_hc_widen_rms`：块内最后写 `inv_rms[row:row+WIDEN_ROWS, 0:1]`，
   `inv_rms` 是 `[N,1]` FP32，8 行正好 **32 B 一次对齐写**。改成 2 行只写 8 B，
   相邻块会写进同一条 32 B 线 —— 竞争。
2. `mix_x_rms_norm`：`hc_pre_fused.py:27` 原注释就是 `T_TILE = 8  # other values miscompare`。
   已经有人试过别的值且数值不一致。
3. `qr_rms_norm_quant`：`pl.store(qr_tile_scale_dq, [out_tg, 0], qr_scale_view)`，
   `qr_scale_view` 是 `[T,1]` FP32，同样靠 8 行凑满 32 B。

要绕开就得把列维也切开、加一级跨块归约（多一个 task + GM 中间量），
按单价 2.3 μs/级 + 0.27 μs/块算，收益会被吃掉大半。**这条路先关掉。**

### 60.6 前半段 AIC 的富余去了哪里

前半段 AIC 的串行链下限 = hc_pre_linear 14.3 + qr_proj 61.6 + qproj 53.0 = 128.9 μs，
再加必须挤进来的 kv_proj 11.4 + kv_score 19.2 = 30.6，合计 159.6。
观测 36.9→280.4 = 243.5 μs，富余 84 μs。逐项超出：

| 任务 | wall | 满核下限 | 超出 |
|---|---:|---:|---:|
| `qproj_matmul` | 112.2 | 53.0 | **+59.2** |
| `hca_kv_score_proj` | 46.4 | 19.2 | +27.2 |
| `kv_proj_matmul` | 24.9 | 11.4 | +13.5 |
| `hc_pre_linear` | 22.7 | 14.3 | +8.4 |
| `qr_proj_matmul` | 65.8 | 61.6 | +4.2 |

`qproj_matmul` 一项占了 70%。第二项 `hca_kv_score_proj`
（`decode_compressor_ratio128.py:34 MM_ROWS=64, MM_COLS=64`，
块数 = `ceil(96/64) × (512/64)` = 2×8 = 16）超出来自 `deps=[late_dep]` 的梯次启动
（wall 46.4 > 最大块 33.4）。MM_ROWS 改 32 在 B16 刚好给 24 块，
但 B24（144 token）会变成 5×8 = 40 块、2 波，反而更差——不是一致的改法，暂不动。

### 60.7 对 0.80× 可达性的重新估计（替代作废的 §58.5）

⚠ 本节初稿把前半段下限写成"AIC 核域下限 159.6"，那是错的：
前半段的 AIV 链上任务被 §60.5 的 32 B 对齐约束钉在 12 核，不可能摊到 48 核，
所以前半段的真实下限是**链上各任务在各自可用核数下的耗时之和**：

widen 28.9（钉死）+ hc_pre_linear 14.3 + mix_x_rms_norm 40.1（钉死）
+ qr_proj_matmul 61.6 + qr_rms_norm_quant 32.6（钉死）+ qproj_matmul 53.0
+ qproj_dequant 50.0 = **280.5 μs**，再加 7 级任务转换约 16 μs（2.3 μs/级）≈ **296 μs**。

于是三段下限：296（前）+ 166.9（attention，已达 97% 核效率）+ 约 140（尾，132.6 + 转换）
= **约 603 μs**。目标是 Native sk=1 的 0.80× = 550.5 × 0.80 = **440.4 μs**。

也就是说，**即使把所有能填的核都填满，仍然差约 163 μs**；
而这三段之间由数据依赖串起来，下限本身不可叠加压缩。
（当前 687.8 → 603 是约 85 μs 的可回收富余，对应比值 1.095×。）
所以 0.80× 不可能只靠"把每段填满核"达到，必须让区段之间重叠，
或者减少总运算量。可能的方向（都还没验证）：

1. **按 token 分片流水**：让先算完的 token 提前进 compressor/attention，
   与后面 token 的 qkv 投影重叠。这是对现有分解方式的结构性改动。
2. **减少 AIV 总量**：attention 的 7751.4 核·μs 是全局最大单项，
   占 AIV 下限的全部。PTO 已比 Native 快，但若要整体降到 440，
   这一段必须也降。
3. **`aiv_vec` 的 39 μs/核超出**（§59.5 第 3 条）是唯一确认的真实计算冗余。

---

## 61. ✗ `QPROJ_WORKERS` 七档加权无效；★ 并暴露 `span_us` 单次重放不可用（2026-09-29）

### 61.1 七档结果：否决

| 档位 | base(24) | w20 | Δ | w32 | Δ |
|---|---:|---:|---:|---:|---:|
| 128K/B4 | 409.00 | 431.75 | +22.75 | 423.50 | +14.50 |
| 128K/B8 | 502.00 | 546.25 | +44.25 | 508.75 | +6.75 |
| 128K/B16 | 693.50 | 673.25 | −20.25 | 651.00 | −42.50 |
| 128K/B24 | 847.00 | 844.00 | −3.00 | 852.50 | +5.50 |
| 8K/B16 | 541.75 | 566.75 | +25.00 | 559.75 | +18.00 |
| 8K/B24 | 689.50 | 706.25 | +16.75 | 711.75 | +22.25 |
| 8K/B32 | 782.00 | 749.00 | −33.00 | 754.75 | −27.25 |
| **8:2 加权** | **624.52** | 633.85 | **+9.33** | 622.23 | **−2.28** |

七档逐 bit 对照全部 PASS。w20 整体更差，w32 的 −2.28 在噪声内。
**`QPROJ_WORKERS` 保持 24，不改。** 变体目录 `variants_qproj_20260929/` 只作取证留存。

### 61.2 ★★ 关键：`span_us` 取单次 profiled 重放，方差盖过了要测的效应

这批数据本身不可信，两个硬证据：

1. **同一配置两次独立运行差很多**：128K/B16 的 w32 在前一批是 663.00，这一批 651.00，
   差 12 μs；w20 是 660.00 → 673.25，差 13.25 μs。
2. **理论上不该受影响的档位摆得最大**：128K/B8 有 24 个 AIC 核全空，
   把 worker 从 24 降到 20 只会让每个 worker 多做一轮、总 wall 略增，
   不可能 +44.25 μs；8K/B32 的 −33.00 同理无法用块数解释。

原因找到了：`device_interval()` 读的是 profiler 那**一次**重放的
`kernel_details.csv`，`span_us` 就是那一次的首尾。
而 `timing.us_p50` 是 100 次采样的中位数——两个量的统计强度差了两个数量级。
我此前建立七档基线时用 `REPEATS=3` 轮压进程间漂移，
却没意识到每轮内部的 `span_us` 只有 **1 个样本**。

### 61.3 修正：同一 profiler 窗口里连采多次、按最大间隔切段、取中位数

`dsv4_hca_compiled_measure.py`：

- `measure_graph_interval(..., profile_replays=N)`：profiler 窗口里连续 `run()` N 次，
  每次之间 `torch.npu.synchronize()`。
- `device_interval(profile_dir, replays=N)`：按 `Start Time(us)` 排序后，
  取全局**最大的 N−1 个 kernel 间间隔**作为切点（重放之间有约 250 μs 的主机派发，
  远大于 kernel 之间的间隔），分成 N 段，每段算 span，返回中位数，
  并附 `span_us_all` / `span_us_min` / `span_us_stdev` / `kernels_per_replay` 以便核对切段是否正确
  （PTO 每次重放应为 2 个 kernel，Native sk=1 应为固定的几十个）。
- `top[].count` 与 `top[].total_us` 一并按 N 归一，`busy_us` 同理。

`dsv4_hca_compiled_case.py` 加 `--profile-replays`，**默认 9**。

**这条口径修正适用于后续所有 HCA 性能对照。** §59/§60 的 pipe 占比与泳道结论不受影响
（那些是比例与核·μs，不依赖单次 span），但 §59.1 里引用的
"PTO span 699.25 / Native 599.5" 这类**单次 span 数字都应视为 ±15 μs 量级的粗值**。

---

## 62. ★★★★ 决定性：七档全为 AIC 受限，AIC 资源下限本身就高于 0.80×（2026-09-29）

### 62.1 新口径七档基线（两侧轮内 9 次重放取中位数）

| 档位 | Native sk=1 | σ | PTO | σ | 比值 |
|---|---:|---:|---:|---:|---:|
| 128K/B4 | 291.75 | 3.91 | 380.00 | 12.91 | 1.302 |
| 128K/B8 | 401.75 | 12.13 | 468.50 | 19.66 | 1.166 |
| 128K/B16 | 539.25 | 8.15 | 641.50 | 13.29 | 1.190 |
| 128K/B24 | 732.00 | 10.54 | 812.00 | 18.05 | 1.109 |
| 8K/B16 | 424.50 | 7.96 | 509.50 | 18.98 | 1.200 |
| 8K/B24 | 512.00 | 12.04 | 650.00 | 16.85 | 1.270 |
| 8K/B32 | 597.75 | 12.11 | 731.00 | 12.19 | 1.223 |

**128K 均值 1.192，8K 均值 1.231，8:2 加权 1.200。**

⚠ 与旧口径的 1.279 相比"变好"了 0.079，**那不是性能变化，是 §61.3 的测量口径修正**：
旧口径每轮只取一次 profiled 重放，而那次紧跟 `reset()`（整份 cache 初态拷贝）之后，
L2 是冷的；多次连采里只有第一次冷，中位数落在暖态上。两侧同样处理，所以可比。

### 62.2 资源下限：不管调度多完美，AIC 都装不进 0.80×

从七档 level-4 泳道按 CoreId 把每块的执行时间分到 AIC（0–23）与 AIV（24–71），
得到"全层总核·μs"，除以核数就是**无视一切依赖与调度、只受核数限制的下限**：

| 档位 | makespan | AIC 核·μs | ÷24 | AIV 核·μs | ÷48 | 资源下限 | 0.80×Native | 差 | 下限比值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128K/B4 | 470.8 | 6690.0 | **278.7** | 6400.8 | 133.3 | 278.7 | 237.8 | −41.0 | 0.938 |
| 128K/B8 | 567.1 | 7914.7 | **329.8** | 10485.2 | 218.4 | 329.8 | 315.0 | −14.7 | 0.837 |
| 128K/B16 | 704.5 | 10780.3 | **449.2** | 16714.5 | 348.2 | 449.2 | 440.4 | −8.8 | 0.816 |
| 128K/B24 | 933.3 | 14871.9 | **619.7** | 21981.1 | 457.9 | 619.7 | 603.8 | −15.8 | 0.821 |
| 8K/B16 | 579.7 | 8427.8 | **351.2** | 10084.6 | 210.1 | 351.2 | 320.4 | −30.8 | 0.877 |
| 8K/B24 | 743.1 | 11710.7 | **487.9** | 16135.4 | 336.2 | 487.9 | 431.0 | −56.9 | 0.906 |
| 8K/B32 | 775.7 | 12750.0 | **531.2** | 18452.7 | 384.4 | 531.2 | 496.4 | −34.8 | 0.856 |

**七档全部 AIC 受限**（AIC 下限是 AIV 的 1.3～2.1 倍），且**每一档的 AIC 下限都已经高于
0.80×Native**。下限比值的 8:2 加权是 **0.858**。

结论分两层，必须分清：

1. **核间流水还有很大空间**：128K/B16 从现在的 1.190 到下限 0.816，
   中间有 **33%** 的调度富余可挖。七档合计 makespan 4774 μs vs 下限 3037 μs。
2. **但只挖调度到不了 0.80×**。要过线，必须同时把 **AIC 总核·μs 降下来**，
   小 batch 尤其吃紧（128K/B4 的下限比值 0.938，几乎没有余量）。

### 62.3 AIC 总量为什么比 Native 多 20%

由 §59 的 pipe 数据：Native 整层每核 AIC 时间 374.2 μs（×24 ≈ 8981 核·μs），
PTO 泳道实测每核忙 449.2 μs（×24 = 10780 核·μs）——**PTO 多 20%**，
而两侧真实 MAC 量相同（2341.6 vs 2320.6 核·μs）。

差在**块内 pipe 重叠度**：

| | 忙时/核 | mte2/核 | mte2 占忙时 |
|---|---:|---:|---:|
| PTO | 449.2 | 252.6 | **56%** |
| Native | 374.2 | 288.2 | **77%** |

Native 的 AIC 核在 77% 的忙时里都在搬运（搬运与 cube 计算高度重叠），
PTO 只有 56%——**PTO 的块内有约 196 μs/核处于"忙但没有任何 pipe 打满"的状态**，
是搬运与计算互相等待。这是 §59.5 判定"核内已到位"之外**真正剩下的核内问题**，
也是唯一能降低 AIC 总量的方向。

### 62.4 候选：`pl.pipeline` 的 stage 深度

全路径审计发现：**HCA 用到的每一个 `pl.pipeline` 都是 `stage=2`**，
包括所有 cube matmul 的 K 循环（proj_a/proj_b/qr_proj/kv_proj/qproj/kv_score/hc_pre_linear
共 14 处）。stage=2 只做二级预取，与 56% 的重叠度吻合。

L1 预算（proj_a 为例）：每级需 x 切片 128×256 BF16 = 64 KiB + w 切片 256×128 BF16 = 64 KiB
= 128 KiB。Mat/L1 是 512 KiB，**stage=3 用 384 KiB 有余量，stage=4 刚好占满**。
已提交 s3 / s4 变体在 128K/B16 与 128K/B4 取证。

### 62.5 ✗ 负结果：wo_a 的 `CachePolicy.BYPASS` 不能去掉

proj_a 读 wo_a（8×4096×1024 BF16 = 64 MiB）时设了 `pl.CachePolicy.BYPASS`。
因为每个 N 列块读各自不重叠的列，层内没有复用，理论上 BYPASS 是对的。
实测去掉后 128K/B16 **从 620.00 变成 646.50（+26.5 μs）**，两者同批提交、可比，
差值是 3.4σ。**保留 BYPASS。**

### 62.6 ⚠ A/B 对照必须同批提交

同一配置（128K/B16 base）在基线批里是 641.50，在 BYPASS 批里是 620.00，
**相差 21.5 μs**——即使轮内已有 9 次重放取中位数，进程间漂移仍有 20 μs 量级。
所以任何变体对照都必须把 base 和变体**放在同一批一起提交**，
不能拿另一批的基线数字去减。

---

## 63. ✗ `pl.pipeline` 的 stage 深度无效；★ 找到 `qr_proj_matmul` 的 3× 权重重读（2026-09-29）

### 63.1 ✗ stage=2 → 3 全部无效（同批对照）

14 处 cube K 循环一起改成 stage=3 直接编译失败：
`ValueError: Mat buffer usage (589824 bytes) exceeds platform limit (524288 bytes)`
（L1 超 64 KiB）。逐个改后，128K/B16 同批结果：

| 变体 | span_us | σ | Δ vs base |
|---|---:|---:|---:|
| base | 641.75 | 14.16 | — |
| proj_a (2 处) | 649.75 | 17.15 | +8.00 |
| proj_b (2 处) | 634.00 | 18.38 | −7.75 |
| qr_proj (4 处) | 646.50 | 13.37 | +4.75 |
| qproj (2 处) | 645.00 | 24.29 | +3.25 |
| kv_proj (2 处) | — | — | L1 溢出，编译失败 |
| kv_score (1 处) | — | — | L1 溢出，编译失败 |

差值全部在 1σ 以内（单侧中位数 SE 约 5～8 μs，差值 SE 约 8～11 μs）。
**stage 深度不是抓手，§62.4 的假设否掉。** 56% 的 mte2 占忙时比例不是预取深度不够造成的。

### 63.2 ★ `qr_proj_matmul` 把 8 MiB 的 wq_a 读成了 24 MiB

`qr_proj_matmul` 算 qr = x[96,4096] × wq_aᵗ[4096,1024]（BF16，NZ 权重）。
MAC 只有 96×4096×1024 = 402.7 M，按 24 核 × 4096 MAC/cycle × 1.85 GHz 算**只需 2.2 μs**，
而实测 wall **65.8 μs**、核·μs 1478.8。纯粹是搬运。

块数怎么来的（`qkv_proj_rope.py:301-302, 317`）：

```
qr_m_groups = pl.min(M_GROUP_LIMIT, pl.max(1, tile_rows // DENSE_M))
pl.spmd(QR_N_BLOCKS * QR_OK * qr_m_groups)
```

T=96 时走 `elif tile_rows < QR_FIXED_SMALL_ROWS(128)` 这一支：
`DENSE_M = QR_FIXED_SMALL_M_TILE = 32`、`M_GROUP_LIMIT = QR_FIXED_M_GROUPS = 3`，
于是 `qr_m_groups = min(3, 96//32) = 3`；`QR_OK = 1`（非 ATOMIC_ADD），
`QR_N_BLOCKS = 1024/128 = 8` → **24 块 = 8 个 N 列块 × 3 个 M 组**。

关键：**M 分组只切行，每一块仍然要读满自己那个 N 列块在全部 K=4096 上的权重**
（128 列 × 4096 × 2 B = 1 MiB）。所以 24 块共读 **24 MiB**，
而 wq_a 一共只有 8 MiB —— **3× 重读**。

对照带宽：
- PTO：24 MiB / 65.8 μs = **382 GB/s**（每核 1 MiB / 61.6 μs = 17 GB/s）
- Native `aclnnMatmulWeightNz`：8 MiB / 约 15 μs = **559 GB/s**

**PTO 读 3 倍的数据、速率还只有 0.68 倍，于是耗时 4.4 倍。**

M 分组不改变归约次序（每个输出元素仍在单块内按固定 K 顺序累加），
所以调它**不动数值**。已提交同批对照：base / m1（1 组，8 块，8 MiB）/
m2（2 组，16 块，16 MiB）/ m1n64（1 组 + N=64，16 块，8 MiB）/
m1n32（1 组 + N=32，32 块，8 MiB）。

⚠ 注意 `QR_N_TILE` 处的既有注释记着 N=32 试过、"in-core 实测 MTE2 占 55.6%、Cube 只有 28.4%"，
但那是在 3 个 M 组下试的，与 m1n32 不是同一配置。

### 63.3 同类检查：proj_a 没有权重重读，但 x 被重读 8 次

proj_a 是 8 个 group 各 8 个 N 块，每块读 wo_a[g,:,n0:n0+128] = 1 MiB，
8×8 = 64 块共 64 MiB = 正好整个 wo_a，**权重无重读**。
被重读的是 x：每个 group 的 8 个 N 块都读同一份 [96,4096] BF16 = 768 KiB，
8× 重读 = 6 MiB/group、48 MiB 合计，但 768 KiB 能常驻 L2，对 HBM 影响小。

按 HBM 口径算：64 MiB / 119.3 μs = **537 GB/s**，
而 Native 的 `aclnnTransposeBatchMatMulWeightNz` 是 64 MiB / 49.02 μs = **1.37 TB/s**。
**proj_a 的权重读取速率只有 Native 的 39%**，这是 proj_a 2.4× 慢的直接原因，
且与 §57 已封死的 tiling 维度（M/N/K）是两件事——那三个维度决定 L0 占用，
这里的问题是 NZ 权重的 DMA 效率。下一轮的核内目标就是它。

---

## 64. ✗ `qr_proj_matmul` 的 M 分组／N 分块已在局部最优；根因是每核 MTE2 速率（2026-09-29）

### 64.1 四个变体全部等于或差于 base（128K/B16，同批）

| 变体 | 块数 | 权重总流量 | 每核流量 | span_us | σ | Δ |
|---|---:|---:|---:|---:|---:|---:|
| base（N=128，3 组） | 24 | 24 MiB | 1 MiB | 636.25 | 14.32 | — |
| m1（N=128，1 组） | 8 | 8 MiB | 1 MiB | 652.25 | 19.90 | +16.00 |
| m2（N=128，2 组） | 16 | 16 MiB | 1 MiB | 635.00 | 25.34 | −1.25 |
| m1n64（N=64，1 组） | 16 | 8 MiB | 512 KiB | 649.50 | 14.27 | +13.25 |
| m1n32（N=32，1 组） | 32 | 8 MiB | 256 KiB | 652.50 | 11.78 | +16.25 |

五个配置全部逐 bit PASS（M 分组与 N 分块都不改归约次序）。

### 64.2 两条判据被同时否掉，指向同一个根因

1. **不是聚合带宽受限**：m1 把权重流量从 24 MiB 降到 8 MiB（1/3），
   耗时反而 +16 μs。若聚合带宽是瓶颈，应当快 3 倍。
2. **不是每核流量受限**：m1n64 / m1n32 把每核流量降到 1/2、1/4，同样没有变快。

把这两条合起来只剩一个解释：**每核的 MTE2 速率被 NZ 切片的连续段长度钉住，
而连续段长度与 N 分块大小成正比，于是"每核少读一半"与"每段短一半"正好抵消。**

NZ 的 fractal 排布是 `[K/16][N/16][16][16]`，固定一个 k-fractal 时相邻 n-fractal 连续，
所以一次载入的连续段 = (N_tile/16) × 512 B：

| N_tile | 连续段 | 实测每核速率 |
|---:|---:|---:|
| 32 | 1 KiB | — |
| 64 | 2 KiB | — |
| 128 | 4 KiB | **17 GB/s** |
| （proj_a 的 wo_a，N=128 但 K=256 且 3D） | 8 KiB | **28～33 GB/s** |
| Native `aclnnMatmulWeightNz`（wq_a） | — | **24 GB/s** |
| Native `aclnnTransposeBatchMatMulWeightNz`（wo_a） | — | **54 GB/s** |

要把连续段做长必须放大 N_tile，但 L0B 限制 `K × N ≤ 32768` 元素（64 KiB BF16）：
N=256 就要 K=128，N=512 要 K=64，而 K 变小会让每核总流量不变、块数变少
（N=512 时只有 2 个 N 块），核用不满。**这个空间是封闭的。**

### 64.3 ⚠ 重要更正：§62.2 的"资源下限"不是物理下限

§62.2 的 AIC 核·μs 是从泳道里把每块的**完整 duration**分到核上算出来的，
而一块的 duration 里包含它自己等 MTE2 的时间。所以那个"资源下限"准确说是
**"在当前每块效率下的下限"**，不是不可突破的物理界。

推论：**如果 NZ 权重的每核读取速率能提上去，AIC 下限本身会跟着下降，0.80× 就重新可达。**
反过来，只做调度（把空转填满）最好到 0.858×，这一点不变。

所以两条轴的分工现在很明确：

- **核间流水**：把 1.200 往 0.858 推，空间 33%，手段是填满空转、消除派发间隙。
- **核内**：把 0.858 这条线本身往下压，唯一确认的抓手是
  **NZ BF16 权重矩阵乘的每核 MTE2 速率**（proj_a 是 Native 的 39%、
  qr_proj 是 68%）。PyPTO 暴露的旋钮（M/N/K 分块、cache policy、pipeline stage）
  已全部试过且封闭，要动就得动 PyPTO 生成 NZ 切片 DMA 描述符的那一层。

### 64.4 本轮已排除的候选汇总（都有同批实测）

| 候选 | 结果 |
|---|---|
| `QPROJ_WORKERS` 24→20/32/48 | 七档加权 +9.33 / −2.28 / — ✗ |
| `pl.pipeline` stage 2→3（5 组分别试） | 全部 ±8 μs 内 ✗ |
| wo_a 去掉 `CachePolicy.BYPASS` | +26.5 μs ✗ |
| `qr_proj` M 分组 3→1／2 | +16.00 / −1.25 ✗ |
| `qr_proj` N 分块 128→64／32 | +13.25 / +16.25 ✗ |
| widen／mix_x／qr_rms_norm 行块 8→小 | 被 32 B 对齐与已知 miscompare 封死 ✗ |

---

## 65. ★★★★★ 静态 CPM 下限判定：0.80× 不可能靠核间流水达到（2026-09-29）

§62 的"资源下限"忽略依赖，§60.7 的"链和"是手算的。关键路径报告里本来就有一个
更权威、更有约束力的量：**static CPM path** ——
"核数无限、调度零损失，只受 happens-before 依赖限制"的延迟下限。
七档都有，直接读出来：

| 档位 | makespan | **静态 CPM** | 资源下限 | 0.80×Native | CPM/Native |
|---|---:|---:|---:|---:|---:|
| 128K/B4 | 469.0 | **353.0** | 278.7 | 233.4 | 1.210 |
| 128K/B8 | 565.0 | **408.0** | 329.8 | 321.4 | 1.016 |
| 128K/B16 | 703.0 | **569.0** | 449.2 | 431.4 | 1.055 |
| 128K/B24 | 931.0 | **734.0** | 619.7 | 585.6 | 1.003 |
| 8K/B16 | 578.0 | **409.0** | 351.2 | 339.6 | 0.963 |
| 8K/B24 | 742.0 | **548.0** | 487.9 | 409.6 | 1.070 |
| 8K/B32 | 774.0 | **558.0** | 531.2 | 478.2 | 0.934 |

**静态 CPM 的 8:2 加权比值 = 1.054**（128K 均值 1.071，8K 均值 0.989）。
七档没有一档的 CPM 低于 0.80×Native，最紧的 128K/B4 是 1.210。

### 65.1 结论

**0.80× 不可能只靠核间流水达到。** 三条下限依次收紧：

| 下限 | 8:2 加权比值 | 含义 |
|---|---:|---|
| 当前实测 | **1.200** | 现状 |
| 资源下限（核数受限，忽略依赖） | 0.858 | 把所有空转填满 |
| **静态 CPM（依赖受限，核数无限）** | **1.054** | 依赖链本身 |
| 目标 | 0.800 | — |

真正的下限是两者中更大的那个，逐档取 max 后**加权约 1.07**。
也就是说：调度侧最多把 1.200 推到约 1.07，**剩下的 0.27 必须从链上每个任务的
自身耗时里拿，或者把链本身改短。**

### 65.2 剩下的两条路（都不是调度）

**A. 压缩链上任务的自身耗时（核内）。** 128K/B16 的观测关键路径 9 个任务：

| 任务 | 链上耗时 | 占 makespan |
|---|---:|---:|
| `hca_unified_attention_aic` | 166.3 | 23.7% |
| `qproj_matmul` | 111.5 | 15.9% |
| `qproj_dequant_rms_nope_rope` | 57.0 | 8.1% |
| `mix_x_rms_norm` | 39.6 | 5.6% |
| `hca_oproj_hc_post` | 36.3 | 5.2% |
| `hca_hc_widen_rms` | 27.1 | 3.9% |
| `qr_proj_matmul` | 25.3 | 3.6% |
| `kv_proj_matmul` | 24.2 | 3.2% |
| `hc_pre_linear` | 21.8 | 3.1% |

attention 已快于 Native（166.9 vs 187.7），不该碰。
`qproj_matmul` 111.5 μs 是第二大，但 §61 证明改 worker 数无效。
唯一确认的核内抓手仍是 §64.3 的 **NZ BF16 权重每核 MTE2 速率**
（proj_a 是 Native 的 39%、qr_proj 68%），而 PyPTO 层面的旋钮已全部试尽，
要动就得动 PyPTO 生成 NZ 切片 DMA 描述符那一层
（用户已授权改 PyPTO/Simpler，见 `pypto-simpler-framework-changes-allowed`）。

**B. 把链改短（结构）。** 现在是"整批 token 走完每一级才进下一级"，
11 级串行。若按 token 分片，让先算完的 token 提前进 compressor/attention，
CPM 能显著下降。代价：任务数翻倍（每级 2.3 μs + 每块 0.27 μs 的派发单价），
以及对现有分解方式的大改。

### 65.3 给用户的判断

以现有分解方式，**0.80× 在七档上都达不到**，这不是调优不够，是依赖链的长度决定的。
可交付的现实目标是 **约 1.05～1.07×**（从当前 1.200 起，纯调度侧的空间）；
要真的过 0.80×，必须从 A（改 PyPTO 的 NZ DMA）或 B（按 token 分片重构）里选一条，
两者都是较大的工程量，需要用户裁决。

---

## 66. ★★★★★ 根因确认：每个 cube 矩阵乘的 L0 ping-pong 都被削掉了，编译器早已报告（2026-09-29）

用户 2026-09-29 要求"先看 `ops-transformer` 里的 incore task 流水是否有可借鉴之处"。
照做之后，Native 的配方与 PyPTO 的 perf hint 两头对上，根因确认。

### 66.1 Native 的配方（`ops-transformer` 的 AscendC 源码）

`experimental/attention/sparse_attn_sharedkv/op_kernel/arch22/sparse_attn_sharedkv_scfa_block_cube.h`：

```
M_SPLIT_SIZE    = 128      N_SPLIT_SIZE = 128
K_L1_SPLIT_SIZE = 256   ← GM→L1 的 K 分块
K_L0_SPLIT_SIZE = 128   ← L1→L0 的 K 分块（两级不同！）
L1_BLOCK_SIZE = 64*512*sizeof(Q_T) = 64 KiB
  bufQPL1 = L1_BLOCK_SIZE * 4 = 256 KiB    bufKVL1 = L1_BLOCK_SIZE * 3 = 192 KiB   （共 448/512 KiB）
L0A_PP_SIZE = L0B_PP_SIZE = 32 KiB → tmpBufL0A/B = 2 × 32 KiB = 64 KiB（满 L0A/L0B）
L0C_PP_SIZE = 64 KiB              → tmpBufL0C  = 2 × 64 KiB = 128 KiB（满 L0C）
```

同步用 7 个 `MTE1_MTE2` flag（`L1_EVENT0..6`）+ 2 个 `M_MTE1`（`L0AB_EVENT0/1`），
逐 slot `SetFlag`/`WaitFlag`，还有反向同步（注释："表示 L1 中的 A 已经被 mte1 消费完"）。
编排是一个 MIX kernel：`if ASCEND_IS_AIC` / `if ASCEND_IS_AIV` 两套代码，
靠 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 握手（4 个 flag 在飞，信用式），
外层 `for (s2LoopIdx < s2LoopEnd + extraLoop)` 是错一拍软流水。

**要点：L1 深度（7）与 L0 深度（2）解耦，L0A/L0B 每片只占一半空间才能 ping-pong。**

⚠ 那个"错一拍软流水"PTO 的 attention **已经有了**
（`hca_unified_attention` 的 `for tick in pl.range(work_count + QK_PRE_LAUNCH)`），
这也正是 PTO 的 attention（166.9 μs）快于 Native 的 `SparseAttnSharedkv`（187.7 μs）的原因。
差距不在 attention，在几个矩阵乘任务上。

### 66.2 PyPTO 的 perf hint 一直在报告同一件事

JIT 的 build output 里有 `report/perf_hints.log`（`PYPTO_LOG_LEVEL` 默认 info 就会写），
之前从没看过。128K/B16 那份里有 **50 条 `PH-MR-001`**，内容全是：

```
MemoryReuse: software pipelining requested depth 2 for pipeline group 0 in Right,
but only 1 of 2 buffers fit (32768 B per stage, 65536 B free)
— stages 1 apart share storage and serialize.
The operand would fit depth 2 on its own, but co-resident buffers /
other pipeline groups over-subscribe the space
```

按源码位置汇总：

| 文件:行 | 任务 | 空间 | 每 stage | 空闲 | 请求→实得 | 条数 |
|---|---|---|---:|---:|---|---:|
| decode_o_proj.py:232 | proj_a | Left | 32768 | 65536 | 2→1 | 6 |
| decode_o_proj.py:232 | proj_a | Right | 32768 | 65536 | 2→1 | 9 |
| qkv_proj_rope.py:249 | qr_proj | Right | 32768 | 65536 | 2→1 | 8 |
| qkv_proj_rope.py:783 | kv_proj | Right | 32768 | 65536 | 2→1 | 8 |
| decode_compressor_ratio128.py:81 | kv_score_proj | Left/Right | 32768 | 65536 | 2→1 | 5+5 |
| decode_o_proj.py:303 | proj_b | Right | 32768 | 65536 | 2→1 | 4 |
| decode_o_proj.py:400 | quant | Vec | 32768 | 188416 | 2→1 | 3 |
| **q_projection.py:98** | **qproj_matmul** | Right | **65536** | 65536 | 2→1 | 1 |
| decode_hca.py:117 | widen | Vec | 16384 | 188416 | 4→3 | 1 |

**每 stage 32768 B 正好是 Native 的 `L0B_PP_SIZE`** —— 说明 `ChooseL0Tile`
选的 L0 分块跟 Native 一样对。L0B 也正好装得下 2 份。
**丢掉 ping-pong 的原因是同一个 task 里并存多个 pipeline group 争这块空间**，
`MemoryReuse` 的跨 group 削减把每个都降到 depth 1，于是
"stages 1 apart share storage and **serialize**" —— MTE1 搬运无法与 MAD 重叠。

**这就是 §62.3 里 mte2 占忙时 56%（Native 77%）的直接机制。**

group 数为什么会多：用户写的 `pl.pipeline(stage=2)` 是 **L1 级**的 K 循环，
而 `AutoTileMatmulL0` 会再为每个 `matmul`/`matmul_acc` 生成一个 **L0 级**的
K 循环（同样 `pipeline_stages=2`）。两层嵌套，副本数相乘
（文档 18-auto_tile_matmul_l0.md 明确说"嵌套 pipeline 仍然走复制路径"）。

所以正确方向与 §63 试的 stage=3 **恰好相反**：应当**减少 group 数**，
把外层 L1 级的 `pl.pipeline(stage=2)` 降成 `pl.range`，
让 L0 的 ping-pong 由 AutoTile 那一层独占空间。已提交同卡 ABBA 对照：
`seqk_all`（14 处 cube K 循环全改）/ `seqk_proja`（只改 proj_a）/
`qproj_n128`（`QPROJ_MM_N_TILE` 256→128，把每 stage 从 65536 B 降到 32768 B）。

`qproj_matmul` 那条最值得单独说：它的每 stage 就是 **65536 B = 整个 L0B**，
**任何深度都不可能双缓冲**。而它是观测关键路径上的第二大项（111.5 μs，15.9% makespan）。

### 66.3 ✗ B（按 token 分片改短依赖链）经计算不成立

§65.2 提的 B 路线在权重受限的矩阵乘上不成立，理由就是 §64 的实测：
`qr_proj` 一类算子**每块都要读它那个 N 列块在全部 K 上的权重**，
与该块处理多少行 token 无关。按 token 分成 S 片，等于把 M 分组乘以 S：

- 每片的 qr_proj 耗时与整份几乎相同（m1/m2/m1n64/m1n32 四个变体实测证明了
  改每核流量不改耗时）；
- 权重总流量乘以 S；
- 任务数乘以 S，再付 2.3 μs/任务 + 0.27 μs/块的派发单价。

也就是说 B 会把最大的几项各算 S 遍。**除非先解决每核搬运速率，B 都是负收益。**
A（修 L0 ping-pong）反而是 B 的前提。

### 66.4 口径再修一次：A/B 必须同卡

§61.3 把轮内改成 9 次重放取中位数，仍然不够。七档复测
`allow_early_resolve` 时 128K/B16 的 Δ 从 −23.75 变成 **+21.00**，
查卡号发现 base/变体分别落在 device 11/13/15 —— **同配置换卡差 45 μs**。

新增 `run_hca_ab_same_card.sh`：一个 task-submit 占一张卡，在**同一张卡**上
按 ABBA…BA 顺序顺序跑各变体（每个变体两个样本，抵消卡内漂移），
各变体各自独立进程与私有 OPP 根。配套 `summarize_hca_ab.py` 汇总。
⚠ `task-submit` 会在命令末尾自动追加 `--device N`，包装脚本收集变体名时
必须在第一个以 `-` 开头的参数处停止，否则它会被当成变体名。

**此前所有跨任务的 A/B 数字（含 §51–§58 的八个否决候选）都是跨卡测的，
结论不可靠，需要用同卡口径重测。**

---

## 67. 同卡口径下的 A 路线实测：`qproj_n128` 是唯一有效项；发现 NZ 档位口径错（2026-09-29）

### 67.1 生成代码给出地面真相：L0 已对齐 Native，差的是 L1 深度

读 JIT 产出的 `ptoas/proj_a_mm.pto`（PTO IR，带源码行溯源），proj_a 的实际分配：

| 层级 | PTO 实际发出 | Native（`ops-transformer`） |
|---|---|---|
| L1 (Mat) | xa 2 份 @131072/196608（各 64 KiB）+ wa 2 份 @0/65536（各 64 KiB）= **256 KiB，深度 2** | **448 KiB，深度 7**（bufQPL1 ×4 + bufKVL1 ×3） |
| L0A (Left) | 2 片 @0/32768，各 32 KiB = 满 64 KiB | 2 × `L0A_PP_SIZE` 32 KiB = 满 64 KiB |
| L0B (Right) | 2 片 @0/32768，各 32 KiB = 满 64 KiB | 2 × `L0B_PP_SIZE` 32 KiB = 满 64 KiB |
| L0C (Acc) | 1 片 128×128 FP32 = 64 KiB | 2 × 64 KiB = 满 128 KiB（dbC） |

**L0A/L0B 的片大小与深度已经和 Native 一致**（`ChooseL0Tile` 选得对），
`wa` 的 `tload` 带 `cache_policy = l2_bypass`。两处差别：
**L1 深度 2 vs 7**，以及 **L0C 没有双缓冲**（Native 有 dbC，PTO 的 drain 暴露在外）。

### 67.2 同卡 ABBA 实测（128K/B16，每变体 2 样本）

| 变体 | 改动 | Δ vs base | 判读 |
|---|---|---:|---|
| `seqk_all` | 14 处 cube K 循环 `pl.pipeline(stage=2)` → `pl.range` | **+25.12** | 明确更差 |
| `seqk_proja` | 只改 proj_a 两处 | −6.25 | 噪声内 |
| **`qproj_n128`** | `QPROJ_MM_N_TILE` 256→128 | **−10.25** | **唯一有效** |
| `qproj_k128` | `QPROJ_PIPE_K_TILE` 256→128 | +4.88 | 减 K 不如减 N |
| `seqk_qprojk128` | 两者合并 | +8.88 | 更差 |
| `m96` | `PROJ_A_ROW_TILE` 128→96 | −0.25 | 中性 |
| `ak128m96` | `A_K_TILE` 128 + m96 | −3.25 | 噪声内 |
| `ak128` | `A_K_TILE` 256→128 | 编译失败 | 见下 |
| `all`（§66 的 early_resolve） | 16 处补 `allow_early_resolve` | −5.62 | 中性偏好 |

**`seqk_all` 的 +25.12 直接否掉 §66.2 提的"减少 pipeline group"方向**：
外层 `pl.pipeline(stage=2)` 是 L1 级预取，去掉它损失的比 L0 争用换回来的多。
结合 67.1 的地面真相，正确方向是**加深 L1**（Native 用 7 级），不是减少 group。

**`qproj_n128` 为什么有效**：`q_projection.py:98` 的每 stage 是
`QPROJ_PIPE_K_TILE(256) × QPROJ_MM_N_TILE(256) × 1 B`(INT8) = **65536 B = 整个 L0B**，
**任何深度都不可能双缓冲**（这是七处 PH-MR-001 里唯一"每 stage 就占满空间"的）。
N 减到 128 后每 stage 32768 B，L0B 能放 2 片。
减 K 同样能到 32 KiB 却反而 +4.88，说明起作用的是 N 方向的块数
（`QPROJ_N_BLOCKS` 从 128 变 256，24 个 worker 上的轮次更细），不只是片大小。

**`ak128` 的硬错误**：`ValueError: Right buffer usage (131072 bytes) exceeds platform
limit (65536 bytes)`。这正是 §57 记的"K=128 → allocator 仍然要 128 KiB"，
现在知道根因：`A_K_TILE=128` 时 L1 stage 2 份 × L0 2 片 = 4 × 32 KiB = 128 KiB，
而旧 PYPTO planner 的 `AllocateMemoryAddr`"只把复用类顺序堆叠、从不细分已释放区域"
（见 `18-auto_tile_matmul_l0.md` 的限制说明与 pypto issue #1908），
所以不会优雅降级，直接报超限。

### 67.3 ★ 口径错误：kernel 侧一直在 nz_mode=1，BF16 权重走的是 ND 分支

从 `proj_a_mm.pto` 的行号溯源发现，编译进去的全部是
`decode_o_proj.py:232–265`，即 **`_proj_a_mm_nd`**，而不是 `_proj_a_mm_nz`（173–217）。

原因：`proj_a_mm = _proj_a_mm_nz if BF16_WEIGHT_NZ else _proj_a_mm_nd`，
而 `BF16_WEIGHT_NZ = WEIGHT_NZ_MODE >= 2`，`WEIGHT_NZ_MODE = envs.VLLM_ASCEND_ENABLE_NZ`。
`run_hca_compiled_case.sh:27` 写的是 `export VLLM_ASCEND_ENABLE_NZ=1` → **1 >= 2 为假**。

而 `--weight-nz-mode 2` 只进 vLLM 的 `additional_config`（主机侧权重存储格式），
report 里 `native_weight_formats` 是 `{wq_a: 29, wq_b: 29, wo_a: 29, wo_b: 29}`
—— **29 = FRACTAL_NZ**。也就是说：**主机把 BF16 权重存成 NZ，kernel 却按 ND 编译**，
只能靠 recast 出私有副本兜住，量到的不是生产路径。
这正是既有约束 `perf-tests-prioritize-nz-mode-2` 警告的情形。

已把 runner 改成 `export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-1}"`
（可外部覆盖），`run_hca_ab_same_card.sh` 支持 `nz1` / `nz2` 标签
（同时设环境变量与 `--weight-nz-mode`，两侧保持一致）。
同卡 `nz1` vs `nz2` 对照已提交。

⚠ 若 nz2 成立，则 §59–§66 的全部数字都是 nz_mode=1 口径下的，需要重建基线。

### 67.4 ✗ 两条方法论教训

1. **A/B 必须同卡**（§66.4 已记）。同卡 ABBA 下 `allow_early_resolve` 是 −5.62，
   而跨卡测出的是 −23.75 与 +21.00 —— 两个都是假的。
2. **不要在任务执行期间编辑 shell 脚本。** bash 是增量读取脚本文件的，
   运行中改动会让它在原字节偏移处解析失败，报
   `line NN: unexpected EOF while looking for matching '"'`。
   三个同卡任务因此都以 exit=2 结束（**各 pass 的数据已落盘、可用**，
   只是最后一行 printf 没执行）。与 `pto-jit-rereads-source-pin-variant`
   是同一类陷阱：JIT 与 bash 都在"运行期"重读源文件。
3. 同卡 ABBA 只 2 个样本时分辨力约 ±7～10 μs：实测 base 自身从第 1 遍的
   627.75 漂到第 4 遍的 642.25（卡内 14.5 μs 漂移）。已给
   `run_hca_ab_same_card.sh` 加 `CYCLES` 支持（`CYCLES=3` → 每变体 6 样本）。

### 67.5 ★★★ 三个有效候选（同卡 ABBA，机制互不重叠）

| 候选 | 改动 | Δ @128K/B16 | 机制 |
|---|---|---:|---|
| **`nz2`** | kernel 侧 `VLLM_ASCEND_ENABLE_NZ` 1→2 | **−22.62** | 口径修正：走 `_proj_a_mm_nz` 而非 ND 分支 |
| **`proja_s3`** | proj_a 的 L1 级 K 循环 `stage` 2→3 | **−12.88** | L1 预取深度，对齐 Native 的 7 级方向 |
| **`qproj_n128`** | `QPROJ_MM_N_TILE` 256→128 | **−10.25** | L0B 片 64 KiB（占满）→32 KiB，恢复 ping-pong |
| `proja_s4` | 同上但 stage=4 | −8.50 | 比 s3 差，L1 384→512 KiB 太挤 |

`nz2` 的分支切换已由 IR 行号溯源确认：nz1 编译 232–259（`_proj_a_mm_nd`），
nz2 编译 186–214（`_proj_a_mm_nz`），两者逐 bit 对照都 PASS。
但 nz2 的两个样本散得大（593.25 / 629.50，差 36 μs），需要更多样本；
无论数字多少，这条都该改——nz_mode=2 是既定的优先口径。

若三者可叠加，128K/B16 约 641.5 − 45 ≈ 596 μs，比值从 1.190 降到约 1.11。
已提交 `qproj_n128` 的三档六样本确认，以及组合变体 `s3_n128`
（proj_a L1 stage=3 + `QPROJ_MM_N_TILE`=128）待在 nz2 下测。

---

## 68. ★★★★ nz2 + 同卡两侧的新验收基线 1.157；`qproj_n128` 与 `proja_s3` 均被推翻（2026-09-29）

### 68.1 ✗ 同卡 2 样本仍然不够：`qproj_n128` 符号翻转

§67.5 里 `qproj_n128` 的 −10.25 μs 是同卡 2 样本测的。加到 **6 样本**（CYCLES=3）
并跨三档复核，结果完全相反：

| 档位 | base | qproj_n128 | Δ |
|---|---:|---:|---:|
| 128K/B16 | 639.62 | 648.38 | **+8.75** |
| 128K/B24 | 817.75 | 828.50 | **+10.75** |
| 8K/B32 | 737.25 | 749.38 | **+12.12** |

三档一致变差，**`qproj_n128` 否决**。
教训：**同卡 ABBA 2 样本的分辨力不足以定符号**，必须 ≥6 样本
（`CYCLES=3`）；2 样本只能用来筛 >30 μs 的粗效应。

### 68.2 ✗ `proja_s3` 在 nz2 下编译失败

组合变体 `s3_n128`（含 proj_a 的 L1 stage=3）在 nz2 下直接报
`ValueError: Mat buffer usage (589824 bytes) exceeds platform limit (524288 bytes)`。
原因：nz2 走 `_proj_a_mm_nz`（186 行）而不是 ND 分支，它的 Mat footprint 更大，
3 个 stage 装不下。**`proja_s3` 的 −12.88 μs 只在 nz1/ND 分支下成立，在生产口径下不可用。**

### 68.3 ★ 新验收基线：nz2 + Native/PTO 同卡

新增的 `run_hca_sides_same_card.sh` 在**一张卡**上按 `native pto pto native` 顺序跑，
Native 开 SuperKernel、PTO 恒 0，两侧 `VLLM_ASCEND_ENABLE_NZ=2` 与
`--weight-nz-mode 2` 一致。七档结果（每侧 2 样本）：

| 档位 | 卡 | Native sk=1 | PTO | 比值 |
|---|---:|---:|---:|---:|
| 128K/B4 | 0 | 290.25 | 364.75 | 1.257 |
| 128K/B8 | 4 | 392.12 | 443.00 | 1.130 |
| 128K/B16 | 5 | 545.00 | 597.50 | **1.096** |
| 128K/B24 | 6 | 720.75 | 805.62 | 1.118 |
| 8K/B16 | 7 | 415.12 | 478.62 | 1.153 |
| 8K/B24 | 9 | 527.75 | 632.38 | 1.198 |
| 8K/B32 | 11 | 594.75 | 714.12 | 1.201 |

**128K 均值 1.150，8K 均值 1.184，8:2 加权 1.157。**

与旧口径（跨卡 + nz1）的 1.200 相比降了 0.043，来源是两件事：
kernel 侧改用 NZ 分支，以及比值不再夹带跨卡差异。
这是目前唯一可信的验收数字。⚠ 每侧只有 2 样本，按 68.1 的教训，
档位级的比值还需要加样本，但两侧同卡已经消掉了最大的误差源。

### 68.4 nz2 下的 perf hint 与 nz1 同构

nz2 的 build output 里 `PH-MR-001` 的位置从 232（ND proj_a）移到 **186（NZ proj_a）**、
从 249 移到 **317（NZ qr_proj）**，其余（`decode_compressor_ratio128.py:81`、
`decode_o_proj.py:303/400`、`q_projection.py:98`、`qkv_proj_rope.py:783`、
`decode_hca.py:117`）完全一样。**§66 的根因在生产口径下同样成立。**

### 68.5 runner 默认已改为 nz=2

`run_hca_compiled_case.sh` 的 `VLLM_ASCEND_ENABLE_NZ` 默认从 1 改为 2，
并写明理由（必须与决定主机侧权重存储格式的 `--weight-nz-mode` 一致）。
此后所有 HCA 测量都在生产口径上。

---

## 69. ★★★ nz2 口径下重算下限：资源下限已压到 0.792（B16），链上两项波次量化占 106 μs（2026-09-29）

在 nz2 口径下重采了 128K/B4、128K/B16、8K/B32 三档 level-4 泳道
（`results/hca_nz2_swimlane_20260929/`），与同卡实测的 Native 对比。

### 69.1 资源下限（核数受限，忽略依赖）

| 档位 | makespan | AIC 核·μs | ÷24 | AIV 核·μs | ÷48 | 资源下限 | 下限/Native | 0.80×Native |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128K/B4 | 462.2 | 5772.5 | **240.5** | 6544.7 | 136.3 | 240.5 | 0.829 | 232.2 |
| 128K/B16 | 708.5 | 10353.8 | **431.4** | 16140.1 | 336.3 | 431.4 | **0.792** | 436.0 |
| 8K/B32 | 806.4 | 12212.8 | **508.9** | 16954.9 | 353.2 | 508.9 | 0.856 | 475.8 |

与 nz1 口径（§62.2）相比，AIC 总核·μs 从 10780.3 降到 10353.8（B16），
下限比值从 0.816 降到 **0.792——第一次低于 0.80**。
B4 从 0.938 降到 0.829，8K/B32 持平 0.856。
**即 128K/B16 这一档的目标已经在核数意义上可达；B4 与 8K 档仍差 3～6%。**

### 69.2 静态 CPM（依赖受限，核数无限）仍是硬约束

| 档位 | makespan | 静态 CPM | Native | CPM/Native |
|---|---:|---:|---:|---:|
| 128K/B4 | 461 | 336 | 290.25 | 1.158 |
| 128K/B16 | 707 | 559 | 545.00 | **1.026** |
| 8K/B32 | 805 | 618 | 594.75 | 1.039 |

CPM 比值从 nz1 的 1.210 / 1.055 / 0.934 变成 1.158 / 1.026 / 1.039。
**依赖链仍然是硬约束，0.80 仍要求把链本身缩短。**

### 69.3 ★ 链上两项波次量化，合计约 106 μs

128K/B16 的观测关键路径（9 个任务，CPM 559 / makespan 707）：

| # | 任务 | 链上耗时 | 块/核 | 满波下限 | 可省 |
|---:|---|---:|---|---:|---:|
| 0 | `hca_hc_widen_rms` | 27.2 | 12/12（32 B 对齐钉死） | 27.9 | — |
| 1 | `hc_pre_linear` | 18.7 | 24/24 | 15.2 | 3.5 |
| 2 | `mix_x_rms_norm` | 40.1 | 12/12（miscompare 钉死） | 39.6 | — |
| 3 | `qr_proj_matmul` | 25.5 | 24/24 | 65.1 | — |
| 4 | `kv_proj_matmul` | 23.5 | 12/12 | 22.7 | — |
| 5 | **`qproj_matmul`** | **112.0** | **24 块 / 20 核 = 2 波** | 51.9 | **60.7** |
| 6 | **`qproj_dequant_rms_nope_rope`** | **87.5** | **48 块 / 43 核 = 2 波** | 42.5 | **45.5** |
| 7 | `hca_unified_attention_aic` | 162.8 | 24/24 | 155.1 | 7.7 |
| 8 | `hca_oproj_hc_post` | 38.0 | 48/48 | 34.4 | 3.6（另有 121.8 core-wait） |

**两者是同一个失效模式**：任务的块数正好等于核域大小（AIC 24 / AIV 48），
而派发时刻有几个核还被别的任务占着（qproj 撞 `hca_kv_score_proj`，
dequant 撞 `hca_cmp_work_gather_0` 的 36 个 AIV 核），于是最后几块排到第二轮，
各自再跑一个完整的块时长 —— **wall 直接翻倍**。

注意 `qproj_dequant` 在 nz1 口径下是 48 块 / 48 核 = 1 波（57.5 μs），
nz2 下变成 43 核 / 2 波（88.1 μs）。同一段代码，只因并发情况变了就多付 30 μs，
**说明"块数 = 核数"这种写法本身是脆弱的**。

正确改法是给块数留余量（少于核域），或把块切得足够细让最后一波便宜。
`QPROJ_WORKERS` 24→20/32/48 在 §61 测过，但那是**跨卡**测的、不可信；
现在用同卡六样本在 nz2 下重测，并加上 `Q_DEQUANT_WORKERS` 48→40：
变体 `qw20` / `dq40` / `qw20_dq40`。

若两项都拿到，128K/B16 约 597.5 − 106 ≈ 492 μs，比值 492/545 = **0.903**。

---

## 70. ★★★ nz2 + 同卡六样本下的候选结果：`early` −20.0、`qw20` −16.6、`dq40` −14.9（2026-09-29）

全部为 nz2 口径、同卡 ABBA、`CYCLES=3`（每变体 6 样本）、128K/B16。

| 候选 | 改动 | Δ | 样本 | 判定 |
|---|---|---:|---|---|
| **`early`** | 16 处 `pl.spmd` 补 `allow_early_resolve=True` | **−20.00** | 6 | 采纳 |
| **`qw20`** | `QPROJ_WORKERS` 24→20 | **−16.62** | 4 | 待满样本 |
| **`dq40`** | `Q_DEQUANT_WORKERS` 48→40 | **−14.88** | 4 | 待满样本 |
| `qw20_dq40` | 两者合并 | −13.00 | 4 | 不如单项，待满样本 |
| `qrproj_s3` | qr_proj 的 L1 级 stage 2→3 | −0.62 | 6 | 否决 |
| `quant_t4` | `QUANT_TOKEN_TILE` 8→4 | 编译失败 | — | 否决（`PartialCodegenError`，quant 函数编不过） |

`early` 的样本分离得很干净：base [604.8, 597.8, 598.2, 599.2, 578.0, 632.0]，
early [579.0, 581.8, 575.0, 578.5, 561.0, 579.0] —— early 的 6 个样本
全部低于 base 的 5 个。

⚠ **同一个改动在 nz1 下只有 −5.62、在 nz2 下是 −20.00。**
口径不对时连候选的排序都会错，这是 §67.3 那个口径 bug 的实际代价。

`qw20` / `dq40` 正是 §69.3 指出的两项波次量化，方向与预测一致，
但实测量级（−16.6 / −14.9）小于满波下限算出的 60.7 / 45.5 ——
说明消掉第二波之后，节省的一部分被别处的 stall 吸收了。

已提交 `early_qw20` / `early_dq40` / `early_qw20_dq40` 三个合并变体。

### 70.1 ★ B 路线的可行窗口：`dequant → attention` 边界

§66.3 判定 B（按 token 分片）在权重受限的矩阵乘上不成立，这一点不变。
但 `qproj_dequant → attention` 这个边界不同：**两者都不重读大权重**
（dequant 是逐元素反量化 + RMSNorm + RoPE，attention 读各 token 自己的 KV 页），
按 token 切片不放大任何流量。

| | 核·μs | 核域 | 下限 |
|---|---:|---|---:|
| `qproj_dequant` | 2041 | 48 AIV | 42.5 |
| `hca_unified_attention_aiv` | 7658 | 48 AIV | 159.5 |
| `hca_unified_attention_aic` | 3722 | 24 AIC | 155.1 |
| **AIV 合计** | **9699** | 48 | **202.0** |

两段当前串行 88.1 + 163.4 = **251.5 μs**，而 AIV 下限只有 202.0，
**重叠可省约 50 μs**。

可行性已确认两点：

1. `simpler/docs/buffer-abi.md` 说明 L2 OverlapMap 做的是**行区间重叠分析**
   （"Stage 2 models only pairs sharing one canonical row-major layout —
   identical strides descending as exact multiples down to 1,
   origins on that layout's lattice"）。`q` 是 `[T, H, HEAD_DIM]` BF16，
   切 [0:T/2] 与 [T/2:T] 的 stride 与原点都落在同一格上，
   **会被判为不重叠**，所以不需要拆成两个 tensor。
2. dequant 本来就带 `tile_base` / `tile_rows` 两个 token 范围参数
   （为多 tile 路径准备的），分两半几乎不用改签名；
   attention 侧要给 `for token in pl.range(worker, t_dim, NUM_QK_CORES)`
   加一个 token 范围。

### 70.2 对 0.80× 的最新估算

128K/B16 当前 597.5（Native 545.00，比值 1.096），0.80× 需要 436.0，即 −161.5 μs。
已识别的可回收量：

| 项 | 预期 |
|---|---:|
| `early` | −20 |
| `qw20` | −17 |
| `dq40` | −15 |
| dequant/attention 按 token 重叠 | −50 |
| 尾段 AIC 富余（proj_a+proj_b 下限 113.9 vs 观测 160.5） | −47 |
| **合计** | **−149** |

→ 约 448.5 μs，比值 **0.823**。也就是说**把已识别的全部拿到，B16 仍差 2.3%**。
再往下只能动 attention 本身（162.8 μs，已在 155.1 的下限上，且已快于
Native 的 187.7），或者做更深的重构。**7 档全部达到 0.80 目前看不到路径**，
但从 1.157 推到约 0.9 是可达的，值得先全部落地。

### 70.3 波次两项的满样本结果与统计力问题

128K/B16，nz2，同卡 ABBA，6 样本：

| 候选 | 中位数 Δ | 均值 Δ | base 样本 | 变体样本 |
|---|---:|---:|---|---|
| `qw20` | **−16.50** | −17.4 | 599.8 619.8 635.0 625.0 632.0 617.8 | 610.0 596.2 585.8 621.0 607.8 604.0 |
| `dq40` | **−14.12** | −4.6 | 623.2 620.2 605.2 620.8 618.8 592.8 | 594.0 598.5 612.2 636.8 616.0 595.8 |
| `qw20_dq40` | −6.00 | −9.8 | 617.5 643.2 598.2 603.5 601.0 619.5 | 600.0 582.8 609.0 613.2 593.8 625.5 |

**⚠ 统计力不足。** 单次测量之间的 stdev 是 13～20 μs，而每个样本**已经是
9 次 profiled 重放的中位数**——也就是说这个方差来自卡内/进程间漂移，
加 `--profile-replays` 没用，只能加 pass 数。
要在 3σ 下分辨 15 μs，需要约 14 个样本（`CYCLES=7`）。

现有 6 样本下：`qw20` 是 1.75σ、`dq40` 约 1.5σ、`qw20_dq40` 不到 1σ。
只有 `early`（−20.00，6 个样本与 base 完全分离）是可靠的。
"合并不如单项"这条也在噪声内，不能据此下结论。

**后续所有候选筛选一律用 `CYCLES=5` 起（10 样本）**，
只有筛出的候选才值得进七档验收。

### 70.4 B 路线窄形式已实现（待测）

按 §70.1 的分析实现了两个变体（`tests/pypto_test/variants_split_20260929/`）：

- **`attn2`**：把 `hca_unified_attention` 按 token 切成 `ATTN_SPLITS=2` 个独立 task，
  用 `for _ah in pl.parallel(ATTN_SPLITS)` 包住原来的 153 行 spmd，
  token 循环改成 `pl.range(_tok_lo + worker, _tok_hi, NUM_QK_CORES)`，
  末尾用 `pl.system.task_dummy(deps=_attn_tids)` 汇合。
  单独切 attention **预期无收益**（dequant 仍是一个 task，前段照样要等它整个完成），
  作为对照用来确认切分本身不引入损失。
- **`attn2_dq2`**：在 `attn2` 基础上给 `q_proj_q_dequant` 加 `row_lo` / `row_hi`
  两个 token 行区间参数（`tile_rows` 保持整个 tile 的行数，尾块判定不变），
  把 `dq_work` 的枚举范围按行区间换算，调用点用
  `for _ds in pl.parallel(DEQUANT_SPLITS)` 拆成 2 段。
  `DEQUANT_SPLITS` 与 `ATTN_SPLITS` 必须一致，`Q_ROPE_T_TILE=8` 下
  T=96 的半点是 48，两侧对齐。

这是 B 路线在**不放大权重流量**的那个窗口上的最小实现。

### 70.5 ★★ 获胜组合 `early + qw20`：−34.75 μs（128K/B16，nz2，6 样本）

| 组合 | Δ |
|---|---:|
| **`early_qw20`** | **−34.75** |
| `early_qw20_dq40` | −27.12 |
| `early_dq40` | +0.50 |

`dq40`（`Q_DEQUANT_WORKERS` 48→40）单独测是 −14.12（1.5σ），
但与 `early` 叠加后变成 +0.50，加进三项组合也把 −34.75 拉回 −27.12。
**`dq40` 否决**——它单独那点收益是噪声，或者与 `early` 争同一份空隙。

`early` + `qw20` 基本可加（−20.00 + −16.50 = −36.5，实测 −34.75）。
128K/B16：597.5 − 34.75 = 562.75，比值 562.75/545.00 = **1.033**（原 1.096）。

### 70.6 ✗ 切分变体第一版编译失败：`@pl.jit` 不支持三元表达式

`attn2` 报 `ParserSyntaxError: Expression must be ScalarExpr or Var with ScalarType,
got MakeTuple with type TupleType`，`attn2_dq2` 报
`UnsupportedFeatureError: Unsupported expression type: IfExp`。

根因是我用了 `t_dim if _ah == ATTN_SPLITS - 1 else (_ah + 1) * _attn_half` 这种
三元表达式。**`@pl.jit` 解析的是源文件 AST，所以源文本里出现三元表达式就会报错，
哪怕条件是 trace 期的 Python 常量。** 同理，Python 级的 `1 if c else 0`
也不能用来做 trace 期选择。

改法：全部换成纯算术 + `pl.min`。段长向上取整再按 `ATTN_SPLIT_ALIGN=8`
（必须等于 `Q_ROPE_T_TILE`）对齐，段上界用 `pl.min((i + 1) * half, t_dim)`，
这样各段既不重叠又覆盖全部 token，且两侧行区间逐段对齐。
新版变体 `attn2b` / `attn2_dq2b`。

---

## 71. B 路线窄形式的完整实现：dequant/attention 按 token 切段 + 显式边（2026-09-29）

### 71.1 为什么必须用显式边，而不能靠自动依赖

`pypto/docs/zh/user/performance/03-dependencies.md` 写明：
**"生产者查找是对 buffer 地址做重叠判定；一个它无法证明不相交的区域，会被当作重叠"**，
并且 **"`pl.parallel` 是断言而不是请求 —— 它不会移除那些边"**。

所以指望 OverlapMap 自动看出 `q[0:48]` 与 `q[48:96]` 不相交是不可靠的
（§70.1 里我据 `simpler/docs/buffer-abi.md` 的行区间分析做的推断过于乐观：
那段描述的是它**能**处理的情形，而"证明不了就当重叠"是兜底行为）。
正确做法是把编译器证明不了的事情**显式说出来**：

- `q` 用 `pl.create_tensor(..., manual_dep=True)` 关掉整个生命期的自动追踪；
- dequant 每段返回自己的 TaskId；
- attention 第 i 段挂 `deps=[..., dq_tids[i]]`。

`q` 的依赖闭环是干净的：**写者只有 `q_proj_q_dequant`
（`qkv_proj_rope.py` 的 577/586/669/682 四处），读者只有 `sparse_attn_hca_tp1`**，
所以关掉自动追踪后只要补这两类边就完整。

### 71.2 改动清单（变体 `variants_split_20260929/split_explicit`）

`deepseek_v4_flash_dspark_perf/qkv_proj_rope.py`：

1. 新增 `DEQUANT_SPLITS = 2`。
2. `q_proj_q_dequant` 加 `row_lo` / `row_hi` 两个 token 行区间参数
   （`tile_rows` 仍是整个 tile 的行数，尾块判定不变）。
3. 它的 `for dq_worker in pl.spmd(...)` 改成 `with pl.spmd(...) as _dq_tid:`
   + `dq_worker = pl.tile.get_block_idx()`，以便取到 TaskId
   （顺带避开 for 形式给出 IterArg 导致的非负不可证问题，与 qr_proj 的既有注释一致）。
4. `dq_work` 的枚举范围按行区间换算（`dq_work` 以 (token 块, head 组) 枚举、
   token 块在外层，所以只要换 start/stop，`tg` 的算法一字不动）。
5. 调用点用 `for _ds in pl.parallel(DEQUANT_SPLITS)` 拆段，
   段长 = `ceil(tile_rows/SPLITS)` 再按 `Q_ROPE_T_TILE` 对齐，
   上界用 `pl.min((_ds+1)*half, tile_rows)`。
6. `dq_tids` 一路返回到 `qkv_proj_rope` 的出口
   （`q_proj_q` → `q_proj_rope` → `qkv_proj_rope`）。

`deepseek_v4_flash_hca/decode_sparse_attn_hca.py`：

7. 新增 `ATTN_SPLITS = 2`、`ATTN_SPLIT_ALIGN = 8`（必须等于 `Q_ROPE_T_TILE`）。
8. `sparse_attn_hca_tp1` / `_short_...` / `_long_...` 三个函数加 `dq_tids` 参数。
9. long 路径的 attention spmd 包进 `for _ah in pl.parallel(ATTN_SPLITS)`，
   deps 加 `dq_tids[_ah]`，token 循环改成 `pl.range(_tok_lo + worker, _tok_hi, ...)`，
   末尾 `pl.system.task_dummy(deps=[_attn_tids[i] for i in range(ATTN_SPLITS)])` 汇合。
10. **交接缓冲 `scores`/`probs`/`values`/`maxima`/`totals`/`ffts` 挪进段内各自分配。**
    它们按 worker 索引，两段并发时 worker 号会重叠——这是切段时最容易漏的一处真实竞争。
11. short 路径不切段，但 `q` 已关自动追踪，所以把 `dq_tids[0]`、`dq_tids[1]` 都挂上。

`deepseek_v4_flash_hca/decode_hca.py`：

12. `q` 加 `manual_dep=True`；接收并转发 `dq_tids`。

### 71.3 踩到的两个坑

- **`@pl.jit` 不支持三元表达式**（§70.6）。段边界只能用 `pl.min` 之类的纯算术表达。
- **TaskId 数组必须用 `pl.array.create(N, pl.TASK_ID)`**，不能用 Python 的
  `[None] * N`——后者会被解析成 `MakeTuple`，报
  `Expression must be ScalarExpr or Var with ScalarType, got MakeTuple with type TupleType`。
  这是 `attn2` / `attn2b` 两版失败的真正原因（一开始误判成三元表达式）。
- 批量改签名时按"签名尾部"匹配会误伤同形的函数：
  `_cmp_query_kv` 的尾部与三个 attention 函数完全一样，被一起加了参数，已撤销。

---

## 72. ★★★★ 落地：`allow_early_resolve` ×16 + `QPROJ_WORKERS` 24→20，七档加权 1.157 → 1.117（2026-09-29）

### 72.1 七档验收（nz2，Native/PTO 同卡 ABBA，每侧 4 样本）

| 档位 | 卡 | Native sk=1 | PTO | 比值 | 基线比值 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 128K/B4 | 2 | 287.00 | 355.25 | 1.238 | 1.257 | −0.019 |
| 128K/B8 | 4 | 401.75 | 429.88 | 1.070 | 1.130 | −0.060 |
| 128K/B16 | 6 | 538.75 | 569.62 | **1.057** | 1.096 | −0.039 |
| 128K/B24 | 7 | 725.88 | 789.50 | 1.088 | 1.118 | −0.030 |
| 8K/B16 | 9 | 414.38 | 448.50 | 1.082 | 1.153 | −0.071 |
| 8K/B24 | 11 | 522.00 | 613.38 | 1.175 | 1.198 | −0.023 |
| 8K/B32 | 13 | 607.62 | 691.50 | 1.138 | 1.201 | −0.063 |

**128K 均值 1.113，8K 均值 1.132，8:2 加权 1.117**（基线 1.157）。
**七档全部改善**，逐 bit 对照全 PASS。

### 72.2 落地的两处改动

1. **16 处 `pl.spmd` 补 `allow_early_resolve=True`**：
   `hca_kv_score_proj`、`hca_softmax_pool`、`hca_state_commit`、`hca_norm_rope_write`
   （`decode_compressor_ratio128.py` ×4）；`hca_hc_widen_rms`、`hca_raw_cache_write`
   （`decode_hca.py` ×2）；三条路径的 `hca_raw_valid` 与 `hca_inverse_rope_sign`
   （`decode_sparse_attn_hca.py` ×6）；`hca_warm_kv_weights`、`hca_warm_wo_a`
   （`weight_warm.py` ×2）；`kv_proj_matmul`、`kv_rms_norm_rope`
   （`qkv_proj_rope.py` ×2）。纯调度标记，不动数值。
2. **`QPROJ_WORKERS` 24 → 20**（`deepseek_v4_flash_dspark/q_projection.py`）：
   `QPROJ_N_BLOCKS = 128`，24 个 worker 只比 AIC 核数少一点，派发时若有核还被占着
   （实测 `hca_kv_score_proj` 会占掉 4 个）就要付两波的钱、wall 翻倍。
   20 块换成 7 轮但恒为一波。

两项单独实测 −20.00 / −16.50，合计 −34.75（基本可加）。

### 72.3 距目标还差多少

0.80× 的目标下，七档加权 1.117 还差 **0.317**。逐档看，最紧的是
128K/B4（1.238）与 8K/B24（1.175）。而 §69.2 的静态 CPM 下限（nz2）是
128K/B4 1.158、128K/B16 1.026、8K/B32 1.039 ——
**128K/B4 现在已经贴到它自己的 CPM 下限（1.238 vs 1.158，只剩 0.08）**，
也就是说那一档几乎没有调度空间了，必须缩短链本身。

### 72.4 十样本确认与 worker 数扫描

`early_qw20` 加到 **10 样本**（`CYCLES=5`）后是 **−47.12 μs**（6 样本时是 −34.75）：

- base：607.0 626.2 603.8 630.2 616.5 622.5 637.5 624.2 623.2 623.2 → 中位数 623.25
- early_qw20：581.8 565.5 590.5 598.0 570.5 558.5 561.8 568.5 600.0 596.2 → 中位数 576.12

两组样本几乎完全分离（只有 600.0 与 603.8 一处交叠），是本轮最干净的结果。

`QPROJ_WORKERS` 扫描（都带 early，6 样本，中位数）：

| W | 轮次 `ceil(128/W)` | span |
|---:|---:|---:|
| 16 | 8 | 593.00 |
| **20** | 7 | **578.38** |
| 22 | 6 | 577.62 |

20 与 22 等价（差 0.76 在噪声内），16 明显更差。保留 20：22 需要 22 个核空闲才是一波，
而实测空闲核数是 20，20 更稳。

⚠ 同卡 A/B 给出的差值（−47.12）与七档验收里的比值改善（B16 1.096→1.057，
折算约 −21 μs）不一致，因为两者的 PTO 基准落在不同卡上（基线 PTO 在 5 号卡读到 597.50，
验收 PTO 在 6 号卡读到 569.62）。**差值信同卡 A/B，比值信同卡两侧验收**，两者各自内部自洽。

---

## 73. B 路线窄形式实测：`split_v2` +32 μs，问题出在每段仍要满核（2026-09-29）

### 73.1 `split_v2` 能编过、逐 bit 正确，但更慢

修掉 §71.3 的两个坑（`pl.Array` 不能做函数参数，报
`Internal error: GetCppType called for Array`；TaskId 数组必须是函数内局部量，
跨函数只能传标量）之后，`split_v2` 编译通过且**逐 bit 对照 PASS** ——
说明 `q` 关自动追踪 + 按段挂显式边这套依赖重写是**正确**的。

但 128K/B16 同卡对照是 **+32.00 μs**（base 中位数 593.50，split_v2 625.50）。

### 73.2 原因：每段仍请求 `NUM_QK_CORES` 块

`hca_unified_attention` 的 spmd 是 `pl.spmd(NUM_QK_CORES=24)`。切成两段之后
两段并发，**合计 48 块抢 24 个 AIC 核 = 两波**——正是 §69.3 / §72.2 刚修掉的
那个失效模式，被我自己在切段时重新引入了。

改法：每段的 worker 数取 `NUM_QK_CORES // ATTN_SPLITS = 12`，
交接缓冲行数（`transfer_rows`）与 token 循环步长同步改成每段核数，
这样两段合计仍是 24 个 AIC 块、48 个 AIV 块，与切段前同形。变体 `split_v3`。

**教训：任何"把一个 task 切成 N 段"的改动，都必须同时把每段的块数除以 N**，
否则块数翻 N 倍、直接撞上核数上限。

### 73.3 ✗✗ B 路线窄形式最终否决，并解释清楚为什么

| 变体 | 每段 worker 数 | Δ vs base（128K/B16 同卡） |
|---|---:|---:|
| `split_v2` | 24（与切段前相同） | **+41.62** |
| `split_v3` | 12（= 24 ÷ 段数） | **+29.75** |

两个都编译通过、逐 bit 对照 PASS，纯粹是更慢。

**§70.1 的收益估算有一个根本错误**：我把 dequant 与 attention 的 AIV 核·μs 相加、
除以 48 得到 202 μs 的"下限"，据此说重叠可省 50 μs。
但**切段之后每段能用的核数也被切了**：

- attention 未切：3722 核·μs ÷ 24 核 = 155 μs
- 切成两段：每段 1861 核·μs ÷ 12 核 = **仍然 155 μs**

两段错开约 44 μs（第二段要等自己的 dequant）启动，结束在 44 + 155 + 44 = 243 μs；
未切是 dequant 88 + attention 163 = 251 μs。**理论收益只有 8 μs**，
而多出来的 task 派发（2.3 μs/任务）、每段独立的交接缓冲、以及段边界上的排队
加起来远不止 8 μs。实测 +30～+42 与这个分析一致。

要真正拿到那 50 μs，两段必须**动态共享整个核池**——段 0 先占满 24 核，
段 1 的数据到了再重新分配。PTO 的 `pl.spmd(N)` 是静态块数，表达不了这种弹性。
**这是 PTO 编程模型的限制，不是调参能解决的。**

**结论：B 路线（按 token 分片缩短依赖链）在宽形式（权重受限的矩阵乘，§66.3
经计算否决）与窄形式（dequant→attention 边界，本节实测否决）上都不成立。**
实现代码保留在 `tests/pypto_test/variants_split_20260929/split_v2`、`split_v3`，
其中"`q` 关自动追踪 + 按段挂显式边"这套依赖重写是正确且可复用的，
将来若 PTO 支持弹性核分配可以直接用。

---

## 74. ✗ `hca_hc_widen_rms` 的 12 块经三次尝试确认无法放开（2026-09-29）

§60.5 判定 `hca_hc_widen_rms` 的行块被 `inv_rms[row:row+8, 0:1]` 的 32 B 对齐钉死。
后来想到一个绕法：**把 `inv_rms` 从 `[N,1]` 补宽成 `[N,8]`，让每行自带 32 B，
相邻行的 4 B 不再共线，行块就能减小**。`inv_rms` 完全是 HCA kernel 内部量
（`decode_hca.py:113` 创建、`hc_pre_fused.py:128/291` 读），接口上可以改。

三次尝试全部被编译器挡回，三个不同的约束：

1. **读侧改 `pl.tile.slice(load([T,8]), [T,1], [0,0])`** →
   `'pto.alloc_tile' op expects result row-major none_box tile row byte size
   (cols * sizeof(dtype)) to be 32-byte aligned, but got 4 bytes`。
   `[T,1]` FP32 的 tile 行只有 4 B，`alloc_tile` 不接受。
2. **写侧改成 `row_expand_mul(pl.tile.full(...), ...)` 广播满行** →
   `pl.col_expand_mul: cannot mix Tensor and Tile arguments`
   （`mean` 是 Tensor 级、`pl.tile.full` 是 Tile 级）；换成 `pl.full` +
   `row_expand_mul` 后又回到第 1 条的 4 B 行问题
   （`[WIDEN_ROWS,1]` 作为算子操作数要被 materialize 成 tile）。
3. **只补宽 tensor、读写写法都不动** →
   `static assertion failed: TLOAD(VecTile, GlobalTensor) only support
   ND2ND/DN2DN/NZ2NZ`。从 `[N,8]` 读 `[T,1]` 是按列跨步，loader 不支持。

三条合起来说明一件事：**要 materialize 一个 `[T,1]` FP32 的 tile，只能从
`[*,1]` 形状的 tensor 直接 load（编译器对这条路径有特判），
而那种 tensor 的行就是 4 B、相邻行必然落在同一条 32 B 线上**，
于是"一个块负责 8 行"是唯一安全的写法。

**`hca_hc_widen_rms`（27.2 μs，链上第 6 大项）确认不可优化。** 同理
`qr_rms_norm_quant` 的 `qr_scale_view`（`[T,1]` FP32，而且是对外接口）也一样。

---

## 75. ★★★★ B 路线终局：想制造的重叠 `allow_early_resolve` 早就在做了（2026-09-29）

§73 把 B 的失败归因为"切段后每段核数也被切了"。那个解释不完整。
采了 base 与切段版的泳道逐 task 对照之后，真正的原因清楚了。

### 75.1 四种块数配置全部更差

| 变体 | dequant 每段块数 | attention 每段块数 | Δ vs base（128K/B16 同卡） |
|---|---:|---:|---:|
| `split_v2` | 48 | 24 | +41.62 |
| `split_v3` | 48 | 12 | +40.12 |
| `split_v4a` | 24 | 24 | +64.88 |
| `split_v4b` | 24 | 12 | +45.25 |

### 75.2 泳道对照：base 已经在重叠了

`base`（128K/B16）：

| 任务 | 起→止 | 块/核 |
|---|---|---|
| `qproj_dequant` | 286.6 → 335.6（49.0） | 48 / 48，1 波 |
| `hca_unified_attention_aic` | **303.5** → 514.8（211.3） | 24 / 24 |

**attention 在 303.5 就起来了，比 dequant 结束（335.6）早 32 μs。**
`allow_early_resolve` 让它在依赖还没完全解决时就派发，块级依赖自然生效——
**我想靠切段制造的那个重叠，框架本来就在做。**

`split_v4b` 的两段 dequant：

| 段 | 起→止 | 块/核 |
|---|---|---|
| 0 | 280.1 → 330.4（50.2） | 24 / 24（另 24 核空转） |
| 1 | 334.8 → 379.0（44.2） | 24 / 24 |

**两段是串行的**，dequant 总跨度从 49.0 变成 98.9 μs。

### 75.3 串行的直接原因：`q` 被 SSA 串起来了

我在调用点写的是 `q, _seg_tid = q_proj_q_dequant(..., q, ...)` ——
段 1 拿段 0 **返回的** `q` 当输入，在 IR 里就是一条实打实的数据依赖，
`manual_dep=True` 管不到它（那只关掉 runtime 的 OverlapMap 追踪，
管不了 SSA 层面的 def-use）。正确写法是让 dequant 只返回 TaskId、不返回 `q`。

但即使修掉这条，B 也没有意义：两段 dequant 都只依赖 qproj、同时就绪，
并发跑就同时结束（各 24 核、各 49 μs），attention 的第 0 段并不会更早拿到数据；
而要让段 0 更早结束就得给它更多核，段 1 又同时在抢。**这个矛盾无解，
而且 base 已经通过 `allow_early_resolve` 拿到了等价的重叠效果。**

### 75.4 ✗✗ B 路线最终结论

**按 token 分片缩短依赖链，在宽形式与窄形式上都不成立，且窄形式的目标收益
本来就已经被 `allow_early_resolve` 实现了。** 四种配置实测 +40～+65 μs。

这也反过来解释了 §72 里 `allow_early_resolve` 为什么值 −47 μs：
它做的正是"让下游在上游还没跑完时就开始"，而这恰恰是 B 想手工实现的东西。
**补全这 16 处标记，等于把 B 想要的重叠一次性拿到了。**

实现代码保留在 `variants_split_20260929/`（v2/v3/v4a/v4b）。其中
"`q` 关自动追踪 + dequant 分段返回 TaskId + attention 按段挂显式边"
这套依赖重写是正确的（逐 bit PASS），将来若有别的用途可直接复用；
⚠ 复用时必须去掉调用点对 `q` 的 SSA 串联，否则分段会退化成串行。

---

## 76. 用同卡口径复核 §57 的 proj_a tiling 否决：结论不变（2026-09-29）

§66.4 记过一条待办：**§51–§58 的八个否决候选全是跨卡 + nz1 口径测的，不可信，
需要用同卡口径重测。** 其中最值得重测的是 proj_a 的 N 分块（它决定块数：
`proj_a_rows × (O_LORA // A_COL_TILE)` = 1 × 1024/N，8 个 group 合计
64 块落在 24 个 AIC 核上 = 3 波）。

| 变体 | 每组块数 | 总块数 | 波次 | Δ vs base（nz2 同卡） | §57 的旧值（跨卡 nz1） |
|---|---:|---:|---:|---:|---:|
| `pn256`（N 128→256） | 4 | 32 | 2 | **+18.75** | +28.8 |
| `pn512`（N 128→512） | 2 | 16 | 1 | 编译失败 | — |

`pn512` 报 `Acc buffer usage (262144 bytes) exceeds platform limit (131072 bytes)`
——L0C 累加器 128×512×4 = 256 KiB，超了一倍。

`pn256` 在正确口径下仍然更差（+18.75，与旧值 +28.8 同号同量级）。
**§57 对 proj_a tiling 的否决在同卡 nz2 口径下依然成立。**

这也说明 §66.4 那条担心（旧否决全部作废）只对**量级小、方向可能翻转**的候选成立
（例如 `allow_early_resolve` 从 −5.62 变成 −20.00、`qproj_n128` 从 −10.25 变成 +8.75）；
对 proj_a tiling 这种量级 20～30 μs、受 L0 容量硬约束的候选，旧结论是可靠的。

---

## 77. 本轮总结：1.157 → 1.117，以及为什么 0.80 在当前分解下不可达（2026-09-29）

### 77.1 交付

| | 8:2 加权比值 |
|---|---:|
| 会话开始（旧口径，跨卡 + nz1） | 1.279 |
| 口径修正后的真实基线（nz2 + 两侧同卡） | 1.157 |
| **本轮落地后** | **1.117** |
| 目标 | 0.800 |

落地的是两处数值中性改动（`allow_early_resolve` ×16 与 `QPROJ_WORKERS` 24→20，
同卡十样本合计 −47.12 μs），七档全部改善、逐 bit 全 PASS。

### 77.2 链上每一项的状态（128K/B16，nz2）

| 链上任务 | 耗时 | 状态 |
|---|---:|---|
| `hca_hc_widen_rms` | 27.2 | 被 `[T,1]` FP32 tile 的 32 B 行规则钉死（§74 三次绕法全失败） |
| `hc_pre_linear` | 18.7 | 下限 15.2，空间 3.5 |
| `mix_x_rms_norm` | 40.1 | `T_TILE=8` 带着既有 miscompare 注记 |
| `qr_proj_matmul` | 25.5 | M 分组/N 分块四种全否（§64） |
| `kv_proj_matmul` | 23.5 | 下限 22.7 |
| `qproj_matmul` | 112.0 | **已由 `qw20` 修掉** |
| `qproj_dequant` | 87.5 | `dq40` 与 early 叠加后无效（§70.5） |
| `hca_unified_attention_aic` | 162.8 | 下限 155.1，且已快于 Native 的 187.7 |
| `hca_oproj_hc_post` | 38.0 | 下限 34.4 |

### 77.3 两条结构性路线的结局

- **A（核内 / 块数与流水）**：`early` + `qw20` 拿到 −47 μs 并落地；其余
  stage 深度、cache policy、L0 分块、M/N 分块、worker 数扫描全部实测否决。
  编译器的 `PH-MR-001` 指出的"每个 cube 矩阵乘的 L0 ping-pong 被削掉"是真的，
  但能调的旋钮都试过了，PyPTO 层面没有更多可动的。
- **B（按 token 分片缩短链）**：宽形式经计算否决（权重重读），
  窄形式四种配置实测 +40～+65 μs，且泳道证明**目标收益已经被
  `allow_early_resolve` 实现了**（base 的 attention 比 dequant 结束早 32 μs 就起来了）。

### 77.4 要继续往下需要的两件事（都超出"调参"范畴）

1. **PyPTO 支持弹性块数**：让一个 task 的块数随派发时刻的空闲核数伸缩。
   本轮三次踩到同一个坑（`qproj_matmul`、`qproj_dequant`、切段后的 attention/dequant）——
   **"块数写死等于核数"这种写法本质上是脆弱的**，并发情况一变就付两波的钱。
   这是运行时/编译器特性，且会影响 CSA 共用的调度器。
2. **重写 `widen → hc_pre_linear → mix_x` 三级**（合计 86 μs，核·μs 只有 1174，
   满核下限约 32 μs）。已设计出 widen 的可行拆法：
   Task A 按 12 行块 × 4 列块 = 48 块做 cast+store 并把列向部分和写进
   各自独占的 `sq_parts[b, 0:8]`（32 B 对齐、块间不共线），
   Task B 12 块读 `[4,8]` 沿列求和后写 `inv_rms`，预计 27.2 → 约 11 μs。
   **但它改变 RMS 平方和的归约次序**（单块顺序累加 16384 个 FP32 → 4 路部分和再合），
   是数值改动而非本轮一直坚持的数值中性优化，需要先确认可接受。

---

## 78. `CMP_GATHER_AIV_WAVE`：同一个"块数写死等于核数"的第四处（2026-09-29）

`decode_sparse_attn_hca.py:83` 的 `CMP_GATHER_AIV_WAVE = 48`，注释原文就是
"一个 AIV 波的块数；长档 gather 的块数按它反推"——又是把块数写死成核数。
nz2 base 泳道实测：`hca_cmp_work_gather_0` 48 块落在 **40** 个空闲 AIV 核上 = 2 波，
wall 50.3 μs 而满波下限只有 26.0，**可省 24.3 μs**。

这一处的分块是纯工作划分（源码注释："划分只决定哪个块搬哪几项，搬运内容与顺序不变"），
**数值中性**。

| 变体 | Δ vs base（128K/B16 同卡，6 样本） |
|---|---:|
| `cw40`（48→40） | **−9.00** |
| `cw36`（48→36） | +0.00 |

−9.00 约 1.3σ，属于边际。本轮的边际候选无一例外在加样本后蒸发
（`dq40` −14.12 → 与 early 叠加后 +0.50；`qproj_n128` −10.25 → +8.75），
所以按 §70.3 定的标准跑 10 样本定论后再决定是否落地。

⚠ 值得注意：预测可省 24.3 μs，实测只有 9。与 `qw20`（预测 60.7、实测 16.5）、
`dq40`（预测 45.5、实测 −14 后归零）一致——**消掉第二波之后省下的时间
大部分被别处的 stall 吸收了**，说明这些任务并不完全在关键路径上。
"满波下限"只能用来找候选，不能用来预估收益。

### 78.1 ✗ `cw40` 十样本定论：+1.12 μs，否决

| 批次 | base 样本 | cw40 样本 | Δ |
|---|---|---|---:|
| 第一批（6 样本） | 597.2 581.5 591.0 607.5 573.2 | 584.8 577.5 579.8 584.2 589.8 574.2 | −9.00 |
| 第二批（8 样本，另一张卡） | 567.5 574.0 597.2 563.5 597.5 553.8 569.8 576.0 | 568.2 575.8 551.5 570.2 561.8 577.2 585.8 590.2 | **+1.12** |

**`CMP_GATHER_AIV_WAVE` 保持 48，不改。**

### 78.2 ★ 本轮的筛选标准，事后验证是对的

本轮总共筛出 6 个"看起来有效"的候选，最终只有 2 个成立：

| 候选 | 6 样本 | 10+ 样本 | 结局 |
|---|---:|---:|---|
| `early`（16 处 `allow_early_resolve`） | −20.00 | **−47.12**（与 qw20 合计） | **落地** |
| `qw20`（`QPROJ_WORKERS` 24→20） | −16.50 | 同上 | **落地** |
| `dq40`（`Q_DEQUANT_WORKERS` 48→40） | −14.12 | 与 early 叠加后 +0.50 | 否决 |
| `qproj_n128` | −10.25（2 样本） | +8.75 / +10.75 / +12.12（三档） | 否决 |
| `qrproj_s3` | −0.62 | — | 否决 |
| `cw40` | −9.00 | +1.12 | 否决 |

规律很清楚：**真实的收益在 6 样本时就已经样本分离**（`early` 的 6 个样本全部低于
base 的 5 个），而边际候选（|Δ| < 1.5σ）在加样本后无一例外地蒸发。
**判据应当是"样本是否分离"，不是中位数差多少。**

这也意味着：§51–§58 那批跨卡 nz1 口径下"±10 μs 以内"的否决，
**其结论大概率仍然正确**——因为那个量级的效应本来就不存在。
只有量级大、或方向可能被口径翻转的（`allow_early_resolve` 从 −5.62 到 −20.00）才需要重测。

---

## 79. ✗ widen 两段拆分实测无效：不用再讨论它的数值代价（2026-09-29）

§77.4 把"重写 `widen → hc_pre_linear → mix_x`"列为需要用户裁决的方向，
理由是 widen 的拆法会改变 RMS 平方和的归约次序。**先做实验、再谈代价**——
结果是收益不存在，代价不必讨论。

### 79.1 实现（`variants_widensplit_20260929/ws4`）

把 `hca_hc_widen_rms` 拆成两个 task：

- **Task A `hca_hc_widen`**：块数 = `widen_blocks(12) × WIDEN_COL_SPLITS(4) = 48`，
  铺满 AIV 核。每块负责一个 (行块, 列组)，做该列组的 cast+store，
  并只累加自己那 1/4 列的平方和，写进**独占的** `sq_parts[unit, 0:8]`
  （行宽 8 个 FP32 = 32 B，块间不共线，所以行块大小不再受 32 B 约束）。
  尾块用的 `tail` 缓冲按列组分区（`[COL_SPLITS * WIDEN_ROWS, HC_DIM]`）避免竞争。
- **Task B `hca_hc_rms`**：12 块，按 split 0→3 顺序把 4 个部分和相加，
  算 rsqrt，按 8 行整块写 `inv_rms`（32 B 对齐）。

编译通过、自一致 PASS。三个 PyPTO 细节：
`pl.rsqrt(..., high_precision=True)` 只接受 Tensor，Tile 形式要用
`pl.tile.rsqrt(tile, tmp=...)`；Tile → Tensor 的写必须走 `pl.store`，
下标写只接受 Tensor 源。

### 79.2 实测：+6.12 μs，否决

| 样本 | |
|---|---|
| base | 589.0 590.5 555.0 583.8 585.0 |
| ws4 | 601.8 580.5 565.5 575.2 603.5 607.5 |

中位数 585.00 vs 591.12 → **+6.12 μs**（4 样本时曾是 −11.12，又是一个翻号的边际候选）。

**widen 拆分否决。** 原因与 §78 记的规律一致：widen 是链头，
把它的 wall 从 27.2 压到理论的 11 μs，省下的时间被下游吸收了——
关键路径上 `hc_pre_linear` 本来就带着 13.7 μs 的 data-wait
（它等的是 widen 的**写**落盘，不只是 task 结束）。

### 79.3 结论：§77.4 的两条"待裁决"路线只剩一条

- **重写 widen 三级**：widen 这一段已实测无效，**不需要用户在数值代价上做裁决**。
  剩下的 `hc_pre_linear`（18.7，下限 15.2）与 `mix_x_rms_norm`（40.1，
  带 miscompare 注记）加起来空间也很小。
- **给 PyPTO 加弹性块数**：仍然是唯一的结构性出路，但 `pypto/` 与 `simpler/`
  是单一共享 checkout（不是 per-worktree），改动会直接影响并行的 CSA 调试 session，
  这条约束是用户定的，需要用户放行。

---

## 80. ★★★★ 直击 PH-MR-001 根因的尝试：算子层做不到，必须改 PyPTO（2026-09-29）

§66 确认了根因：每个 cube 矩阵乘的 L0 ping-pong 被削到 depth 1，
编译器提示是 "**other pipeline groups over-subscribe the space**"。
一直没试过的正解是**减少一个 task 里的 pipeline group 数**。

### 80.1 group 从哪来：剥离写法

`proj_a`（`decode_o_proj.py:205`）与 `hca_kv_score_proj`
（`decode_compressor_ratio128.py:90`）都用**剥离写法**：把 k=0 那一步单独写成
`pl.matmul`，再用 `pl.pipeline` 做后续的 `matmul_acc`。
`AutoTileMatmulL0` 会给**每个** matmul 调用各生成一个 L0 K-loop
（`pipeline_stages=2`），于是：

- `proj_a`：2 个 group × 2 个 slot × 32 KiB = 128 KiB > 64 KiB 的 L0B
- `hca_kv_score_proj`：两个累加器各剥一次 = **4 个 group**（实测 hint 报了 5 条）

`AutoTileMatmulL0` 的文档明确写了另一种规范形式——**谓词写法**：
单个 4 操作数 `tile.matmul_acc(acc, lhs, rhs, init_cond)`，没有分支也没有 phi。
合成一个 group 后 2 个 slot 正好装下 64 KiB，而且 K 的累加次序完全不变、**数值等价**。

### 80.2 ✗ 三次尝试，三个不同的拒绝

| 种子写法 | 报错 |
|---|---|
| `pl.create_tensor([1, M, N], FP32)` | `NZ layout currently supports only matmul operand loads (target_memory=pl.Mem.Mat), got Vec` |
| 同上（kv_score） | `Verification failed after 'InferTileMemorySpace'` |
| `pl.full([1, M, N], FP32, 0.0)` | `tile.batch_matmul_acc requires argument 0 to live in Acc memory, but it is in Vec memory. No target has any data path into Acc` |

第三条是要害：**累加器必须驻留 Acc（L0C），而张量层能造出来的种子只会落在 Vec，
两者之间没有数据通路。** 文档里说的"iter-arg 初值为 Acc-resident 的
`tile.create([m, n], dtype, target_memory=Acc)` 种子"是 `AutoTileMatmulL0`
在 **tile 层自己生成**的东西，张量层 API 表达不出来。

再者，`AutoTileMatmulL0` 识别"前端规范 split-K"需要
"一个 `tile.create` 全输出累加器占位值 + 循环携带它完成 K 归约 + 一个 2D 输出 store"
这三段相邻；而 proj_a 的结果流向 `pl.assemble`（不是 2D store）、
操作数还是 3D（`wo_a` 带 group 轴，走 batch_matmul），两条都不匹配。

### 80.3 ★ 结论：一个精确的 PyPTO 需求

**PH-MR-001 指出的根因在算子层无法规避——剥离写法是张量级 API 下的唯一选择。**
要消掉它，需要 PyPTO 做下面任一件事：

1. **让 `AutoTileMatmulL0` 识别剥离写法并自动合并**成单个带谓词的 K-loop
   （它已经能识别"peel 写法"的 split-K 三元组，但只在 M/N 切分路径上；
   这里需要的是在**不切 M/N**、只切 K 的路径上也做同样的合并）；
2. 或者**允许张量层声明 Acc-resident 的累加器种子**，让作者能自己写谓词形式。

这比 §77.4 里"给 PyPTO 加弹性块数"具体得多，改动面也小得多——
它只动 `AutoTileMatmulL0` 这一个 pass，不碰运行时调度器，
因此**不影响并行的 CSA 调试 session**（那条约束针对的是共享运行时）。

预期收益：把每个 cube 矩阵乘的 L0 ping-pong 从 depth 1 恢复到 2，
按 §62.3 的 pipe 数据，PTO 的 AIC 忙时里只有 56% 有 mte2 在跑（Native 是 77%），
这 21 个百分点正对应搬运与计算无法重叠。AIC 总量 10353.8 核·μs 若能降到
Native 的 8981，B16 的资源下限从 431.4 降到 374，比值下限从 0.792 到 **0.686**。

---

## 81. ★★★★★ 真正的出路：换 `memory_planner=PTOAS`，而且不用改 PyPTO（2026-09-29）

§80 的结论是"PH-MR-001 在算子层无法规避，需要改 PyPTO"。**那个结论是错的。**
`LowerPipelineToSlots`（pypto 文档 30）写得很清楚：

> `pl.pipeline(N, stage=F)` 表达的是乒乓缓冲的诉求。`LowerPipelineLoops` 用**复制**
> 来兑现……本 pass 用 `pl.MemRef(name, slots=F)` 表达同一个意图，循环只保留**一份**
> 循环体……**而且该流水线本就跳过 MemoryReuse。**
> ……**自门控于 `memory_planner=PTOAS`。**

也就是说：**PTOAS planner 下根本不会产生 `pipeline_membership`，
也就不会有"多个 group 争 L0"的 PH-MR-001**，而且文档还说
"`DSA_RP` 与 `PTOAS` 已按实际生命周期放置缓冲"、"dbC=2 是自动的"
（正是 §67.1 里 Native 有、PTO 没有的 L0C 双缓冲）。

### 81.1 怎么切：一个调用点参数，不碰 PyPTO

`pypto.torch.register(kernel, name, *, constexpr=None, config: CompileOptions | None)`，
而 `CompileOptions` 有 `memory_planner: MemoryPlanner | None`。
所以在 `native_adapter.py` 的 register 处加一个 config 即可：

```python
return cls(pypto.torch.register(
    decode_hca_tp1_layer_test, "dsv4_hca::attention",
    config=CompileOptions(platform="a2a3", memory_planner=MemoryPlanner.PTOAS),
))
```

⚠ **`CompileOptions.platform` 默认是 `"a2a3sim"`（模拟器），必须显式写 `"a2a3"`。**
（另一条等价路径是 `with PassContext([], memory_planner=...)` 包住编译调用，
但 PassContext 是编译期动态作用域，register 这条更稳。）

**`DSA_RP` 直接不可用**：`DSA-RP could not find a placement for 'proj_a_mm'
within the on-chip memory capacities: canonical greedy found no capacity-fitting placement`。

### 81.2 PTOAS 的阻塞项：8 → 3，且修法就是 Native 的配方

首次切 PTOAS 有 **8 个函数编不过**。三次迭代后降到 3 个：

| 迭代 | 改动 | 修掉的函数 |
|---|---|---|
| 初始 | 仅换 planner | — （8 个失败） |
| 1 | `A_K_TILE` 256→128、`QR_K_TILE` 256→128 | `proj_a_mm`、`proj_a_mm_0`、`qr_proj_matmul`、`qr_proj_matmul_0` |
| 2 | `PROJ_A_LARGE_N_TILE` 256→128 | `proj_a_mm_1` |
| 剩余 | — | `qr_rms_norm_quant`、`hca_short_attention_pack`、`hca_unified_attention` |

**报错本身就说明了方向是对的**：

```
proj_a_mm      : left  overflow, requires 1048576 bits (128 KiB) while 524288 (64 KiB)
qr_proj_matmul : right overflow, requires  786432 bits ( 96 KiB) while 524288 (64 KiB)
proj_a_mm_1    : right overflow, requires 1048576 bits (128 KiB) while 524288 (64 KiB)
```

PTOAS **强制**每个 L0 片只占容量的一半，这样 2 个 slot 才装得下——
**这正是 Native 的 `L0A_PP_SIZE = L0B_PP_SIZE = 32 KiB`（§66.1）**。
换句话说：PTOAS 在逼着 kernel 用 Native 的分块，而 PYPTO planner 只是
默默把 ping-pong 削掉、报一条 perf hint 就算了。

注意 `A_K_TILE=128` 在 PYPTO planner 下是**硬错误**
（§67.2 的 `ak128`：`Right buffer usage 131072 > 65536`，根因是旧 allocator
不细分已释放区域），**在 PTOAS 下却是正解**——两个 planner 对同一个分块的判断相反。

### 81.3 剩余 3 个阻塞项与修法方向

| 函数 | 错误 | 方向 |
|---|---|---|
| `qr_rms_norm_quant` | `pl.set_validshape cannot narrow a tile view`；改成 `pl.tile.slice(..., valid_shape=)` 后变成 `failed to legalize` | `qkv_proj_rope.py:488` 的 `qr_tile_scale_dq` 是 `pl.reshape` 出来的视图。应当在**源** tile（`qr_scale_quant_row`，形状 `[1, T_TILE]`）上先收窄再 reshape |
| `hca_short_attention_pack` | `vec overflow, requires 1832960 bits while 1572864 bits available`（229 KB > 192 KB UB） | 8K 短路径的 UB 占用要缩；PTOAS 的 slot 分配比复制路径更保守 |
| `hca_unified_attention` | `'pto.tmov' op expects a supported tmov address-space pair`，位置 `decode_sparse_attn_hca.py:1194` | 该行附近有一处 PTOAS 不支持的跨内存空间 move，需要换等价写法 |

### 81.4 预期收益

按 §62.3 的 pipe 数据：PTO 的 AIC 忙时里只有 **56%** 有 mte2 在跑，Native 是 **77%**——
这 21 个百分点正是搬运与计算无法重叠。若 AIC 总量从 10353.8 降到 Native 的
8981 核·μs，128K/B16 的资源下限从 431.4 降到 374，**比值下限从 0.792 到 0.686**。
再叠加 PTOAS 自动开的 L0C dbC=2（Native 有、PTO 现在没有），还有额外空间。

**这是目前唯一一条既有明确机制、又不需要改 PyPTO、也不影响 CSA 调试 session 的路线。**
变体留在 `tests/pypto_test/variants_planner_20260929/ptoas_k128b`，
从那里接着修剩下 3 个函数即可。

### 81.5 阻塞项推进到 2 个（8 → 4 → 3 → 2）

| 迭代 | 改动 | 修掉的函数 | 剩余 |
|---|---|---|---:|
| 0 | 只换 planner | — | 8 |
| 1 | `A_K_TILE` 256→128、`QR_K_TILE` 256→128 | proj_a_mm、proj_a_mm_0、qr_proj_matmul、qr_proj_matmul_0 | 4 |
| 2 | `PROJ_A_LARGE_N_TILE` 256→128 | proj_a_mm_1 | 3 |
| 3 | `qr_rms_norm_quant` 改用 `pl.store(tile, off, out, [valid_rows, 1])` | qr_rms_norm_quant | **2** |

第 3 步值得单记：`pl.store` 的第三个位置参数 `shapes` 可以直接限定写入范围，
**根本不需要给 tile 打 valid 标记**。原来的
`pl.set_validshape(view, ...)` 在 PTOAS 下被拒（视图的 valid 写在类型里），
改成在切片处收窄又 `failed to legalize`，而 `pl.store(..., shapes=)` 一次通过。

变体：`tests/pypto_test/variants_planner_20260929/ptoas_k128c`。

### 81.6 最后 2 个阻塞项：都要动最热的代码

**`hca_unified_attention`** —— `'pto.tmov' op expects a supported tmov address-space
pair`，位置 `decode_sparse_attn_hca.py:1194`。那里是
`pl.gather_row(kv_l1, raw_kv / cmp_work_kv, ...)`，`kv_l1` 是
`pl.create_tile(..., target_memory=pl.MemorySpace.Mat)` 的**环形缓冲**
（`QK_TRANSFER_SLOTS` 个槽），两个分支都走 GM→Mat 的 `gather_row`。
`pl.gather_row` 的文档写明它"always reads from global memory"，
PYPTO planner 能把它降到 Mat，PTOAS 的 tmov 不支持这个地址空间对。

绕开它意味着换掉整个 KV 入 L1 的机制——而那个环形缓冲正是 attention 的手写软流水
（配合 `QK_PRE_LAUNCH` 的错拍，§66.1 里说这是 PTO attention 快于 Native 的原因）。
**这是一次对最热代码的结构性重构，不能顺手做。**

**`hca_short_attention_pack`** —— `vec overflow, requires 1832960 bits while
1572864 bits available`（229 KB > 192 KB UB）。8K 短路径的 UB 占用要缩约 37 KB。
PTOAS 的 slot 分配比复制路径保守，同样的代码在 PYPTO 下装得下。

### 81.7 续做指引

从 `variants_planner_20260929/ptoas_k128c` 接着改。两个建议：

1. 先只让 **128K 路径**（`_long_sparse_attn_hca_tp1`）在 PTOAS 下通过——
   `hca_short_attention_pack` 属于 8K 短路径，可以先用一个编译期开关把短路径
   排除在 PTOAS 之外（两条路径在 `sparse_attn_hca_tp1` 里由
   `pl.tensor.dim(cmp_block_table, 1) <= CMP_PAGES_PER_WORK` 分派），
   拿到 128K 档的数字再回头处理 8K。
2. `gather_row` 那处，先确认 PTOAS 是否支持 **GM→Vec 的 gather_row + Vec→Mat 的
   `tile.move`** 两步走（`AutoTileMatmulL0` 文档里的 "Vec 左操作数预存" 就是
   `tile.move(lhs, target_memory=Mat)`，说明 Vec→Mat 这一跳是支持的）。
   代价是 `[128, 512]` BF16 = 128 KB 过一次 UB，要看 UB 预算是否还够。

---

## 82. PTOAS 路线推进到最后 1 个阻塞项，并确认它卡在哪（2026-09-29）

### 82.1 阻塞项 8 → 1

| 迭代 | 改动 | 修掉 | 剩余 |
|---|---|---|---:|
| 0 | 只换 `memory_planner=PTOAS` | — | 8 |
| 1 | `A_K_TILE`、`QR_K_TILE` 256→128 | proj_a_mm、proj_a_mm_0、qr_proj_matmul、qr_proj_matmul_0 | 4 |
| 2 | `PROJ_A_LARGE_N_TILE` 256→128 | proj_a_mm_1 | 3 |
| 3 | `qr_rms_norm_quant` 改用 `pl.store(tile, off, out, [valid_rows, 1])` | qr_rms_norm_quant | 2 |
| 4 | 强制只走长路径（仅取证，8K 短路径先排除） | hca_short_attention_pack | **1** |

变体链：`ptoas` → `ptoas_k128` → `ptoas_k128b` → `ptoas_k128c` → `ptoas_long`。

### 82.2 最后一个：`hca_unified_attention` 的 KV 入 L1

报错是 `'pto.tmov' op expects a supported tmov address-space pair`，
位置在 `pl.gather_row(kv_l1, raw_kv / cmp_work_kv, ...)`。三次探查：

1. **不是子区域偏移的问题。** 把 `QK_PRE_LAUNCH` 置 0（槽数变 1、目标偏移恒为 0、
   写的是整片），**仍然同样报错**。所以 PTOAS 拒的是 `gather_row` 这种
   **DPS 形式的 GM→Mat 写**本身。
2. **不是 GM→Mat 本身不行。** 同一段里 `query = pl.load(q_flat, ...,
   target_memory=pl.MemorySpace.Mat)` 在 PTOAS 下是通过的。区别在
   `pl.load` 新建 tile，而 `gather_row` 往已有 tile 里写。
3. **`pl.tile.assemble`（Mat→Mat 子区域）也不行**，同样的 tmov 报错。

于是尝试把 Mat 环形缓冲换成 `pl.range(..., init_values=...)` 的**移位寄存器**
（`QK_PRE_LAUNCH=2` 意味着 PV 在第 t 拍消费第 t−2 拍的数据，只需携带 2 个值），
结果 `Verification failed after 'ConvertToSSA'`，6 条
`Variable '...' used outside its defining scope`，点名的是 `kv_p2`（iter_arg）
与 `kv_seed`（`pl.create_tile(..., target_memory=Mat)`）。

对照同一函数里**能工作**的 tile 携带例子（`for vec_tick, (m_iter, l_iter,
left_iter, right_iter) in pl.range(..., init_values=(...))`，
`decode_sparse_attn_hca.py:1233`）：它携带的全是 **Vec** tile。
**结论：Mat 驻留的 tile 不能作为循环携带值**——它的生命期绑在定义作用域上。
而 `pl.matmul` 的 B 操作数又必须是 Mat（`AutoTileMatmulL0` 文档：
"对于自动 tiling，右（B）操作数必须是 Mat"），所以这条路封死。

### 82.3 结论

**`hca_unified_attention` 要在 PTOAS 下编过，只有两条路，都超出"改算子"的范畴：**

1. 重新设计 KV 进 L1 的机制，不用持久的 Mat 缓冲——但那个缓冲正是 attention 的
   手写软流水（QK 写第 t 槽、PV 读第 t−2 槽），§66.1 里说它是 PTO attention
   比 Native 快 25 μs 的原因。改掉它等于放弃这个优势去换 L0 ping-pong，
   净收益未知。
2. 让 PTOAS 支持 `gather_row` 的 GM→Mat 下降（PyPTO 改动，按用户
   2026-09-29 的要求需要先商量）。

**PTOAS 路线到此为止：非 attention 的部分全部就绪，attention 这一个函数挡住了整体。**
由于 planner 是整个 kernel 编译级的设置，一个函数编不过就整体编不过，
所以拿不到"其余任务在 PTOAS 下变快多少"的数字。

变体全部留在 `tests/pypto_test/variants_planner_20260929/`
（`ptoas_k128c` 是非 attention 部分全部修好的版本，`ptoas_long` 再叠加强制长路径，
`ptoas_shift` 是移位寄存器的失败尝试）。

### 82.4 第三次尝试：展平循环、每拍新建 Mat 片——仍失败，但换了一种错

既然 Mat 片不能跨拍存活，就把错拍也去掉：`QK_PRE_LAUNCH = 0` 之后
PV 在同一拍消费同一片，两个 `if` 恒真、循环体可以展平，
KV 每拍用 `pl.load(..., target_memory=Mat)` 新建一片，**完全不需要持久缓冲**。
分支用 `pl.yield_` 汇合（本文件里既有的 tile 分支写法）。

结果是第三种错误：

```
'pto.tmov' op expects A2/A3 non-mat tmov to use matching src/dst shapes
```

位置在 `pypto/python/pypto/language/parser/ast_parser.py:3837`（编译器内部生成的 move），
最可能是 `pl.yield_` 合并两个 Mat 片时发出的 phi move。

### 82.5 ✗ PTOAS 路线收尾

三次尝试、三种不同的拒绝，指向同一件事：

| 尝试 | 写法 | 错误 |
|---|---|---|
| 1 | `pl.gather_row` 往 Mat 环形缓冲写（原样，含"写整片"探针） | `unsupported tmov address-space pair` |
| 1' | `pl.load` 到 Mat 片 + `pl.tile.assemble` 进环形缓冲 | 同上 |
| 2 | `pl.range(init_values=...)` 携带 Mat 片做移位寄存器 | `ConvertToSSA` 失败，Mat 片 `used outside its defining scope` |
| 3 | 展平循环、每拍新建 Mat 片、分支 `pl.yield_` 汇合 | `non-mat tmov to use matching src/dst shapes` |

**PTOAS 对 Mat 的处理远比默认 PYPTO planner 严格**：不能对 Mat 做 DPS 写、
Mat 片不能跨迭代存活、Mat 片的分支 phi 也受限。而 `hca_unified_attention`
的设计重度依赖 Mat（KV 环形缓冲、query、probability 都驻留 Mat，
且 `pl.matmul` 的 B 操作数必须是 Mat）。

**所以 PTOAS 路线在不重写 attention 的前提下走不通**，而 planner 是 kernel
编译级设置，这一个函数编不过就拿不到任何 PTOAS 下的数字。
非 attention 部分的修改（`ptoas_k128c`）是有效且可复用的，留在变体目录里。

**这一轮到此为止：七档加权 1.117，距 0.80 差 0.317，没有找到不越界的路径。**

---

## 83. ✗ `enable_pypto_l0c_double_buffer`：旧 allocator 装不下（2026-09-29）

§82 之后还剩一个绕开 attention 阻塞、仍能拿到 PTOAS 一半收益的可能：
**在默认 PYPTO planner 下单独打开 L0C 双缓冲**。这是 §67.1 里
Native 有（`ops-transformer` 的 `L0C_PP_SIZE = 64 KiB`，两片共 128 KiB 满 L0C）
而 PTO 没有的那一项；没有它，`AutoTileMatmulL0` 的 roofline 是
`wall ≈ max(C_load, C_mad) + C_drain`，FIXPIPE 的 drain 完全暴露。

### 83.1 怎么开

这个 flag **只能经 `PassContext` 生效**——`CompileOptions` 里没有对应字段
（`pypto/python/pypto/ir/compile.py:177` 有 `enable_pypto_l0c_double_buffer` 这个
kwarg，但 `CompileOptions.as_compile_kwargs()` 不产出它）。
而 PyPTO 的 JIT 在**首次调用时**编译，所以 PassContext 必须在那时生效。

已给 `dsv4_hca_compiled_case.py` 加 `--pypto-dbc`，用 `contextlib.ExitStack`
在 `compiled_call` 里叠一层 `PassContext([], enable_pypto_l0c_double_buffer=True)`；
`run_hca_ab_same_card.sh` 加 `dbc` 标签（与 `nz1`/`nz2` 同一机制，只切编译开关、
源码用生产版）。

### 83.2 实测：编译失败

```
Right buffer usage (131072 bytes) exceeds platform limit (65536 bytes)
Check failed: limit == 0 || used <= limit
  at pypto/src/ir/transforms/allocate_memory_addr_pass.cpp
```

注意溢出的是 **Right（L0B）而不是 Acc（L0C）**。原因在
`AutoTileMatmulL0` 的 chooser：有没有 dbC 用的是**两个不同的 roofline**

- 无 dbC：`wall ≈ max(C_load, C_mad) + C_drain`
- 有 dbC：`wall ≈ max(C_load, C_mad, C_drain) + min(compute, C_drain) / T`

于是开了 dbC 之后 `ChooseL0Tile` 选出**另一个更大的 (m, n, k)**，L0B 装不下。
而旧 PYPTO allocator "只是把复用类顺序堆叠、从不细分已释放区域"
（`18-auto_tile_matmul_l0.md` 的限制说明 + pypto issue #1908），不会优雅降级。

这与 §67.2 的 `ak128` 是**同一个报错、同一个根因**：
`A_K_TILE=128` 当初也是 `Right buffer usage 131072 > 65536`。
**旧 allocator 是这两件事共同的拦路石**，而 `DSA_RP` / `PTOAS`
"已按实际生命周期放置缓冲"本可以处理——但那两个 planner 又各自被别的东西挡住
（DSA_RP 找不到 `proj_a_mm` 的放置方案，PTOAS 卡在 attention 的 Mat）。

### 83.3 本轮可用手段至此穷尽

| 层面 | 手段 | 结果 |
|---|---|---|
| 算子·块数/分块 | 9 个候选（worker 数、L0/L1 分块、cache policy、pipeline stage、行块…） | 2 个落地（−47 μs），7 个实测否决 |
| 算子·结构 | B 路线：按 token 分片缩短依赖链 | 四种配置实测 +40～+65 μs，且目标收益已被 `allow_early_resolve` 实现 |
| 编译期开关 | `memory_planner=PTOAS` | 非 attention 部分 8→1，attention 的 Mat 用法四种写法全被拒 |
| 编译期开关 | `memory_planner=DSA_RP` | `proj_a_mm` 找不到片上放置方案 |
| 编译期开关 | `enable_pypto_l0c_double_buffer` | L0B 溢出，旧 allocator 装不下 |

**七档加权 1.117，距 0.80 差 0.317。在不改 PyPTO、不重写 attention 软流水的前提下，
没有找到进一步的路径。**

### 83.4 ✗ 第六次重写 attention：剥出 raw 块也不行

§82.4 的 `ptoas_flat` 死在 `pl.yield_` 合并两个 Mat 片的 phi 上。
注意到 `work_count = 1 + cmp_blocks`、raw 块恒为最后一拍，于是把它**剥出循环**，
循环体内不再有分支、也就没有 Mat phi（变体 `ptoas_peel`）。

**仍然是 `'pto.tmov' op expects A2/A3 non-mat tmov to use matching src/dst shapes`。**

所以上一版的失败也不是 phi 造成的。这段代码在 PTOAS 下还有别的构造会生成非法 move，
报错位置是编译器内部的 `pypto/python/pypto/language/parser/ast_parser.py:3837`，
无法对应到源码的具体哪一行；可疑的是
`pl.tile.move(probability, target_memory=Left)` 与
`pl.tile.extract(kv_new, ..., target_memory=Right)` 这两类跨空间搬运——
而它们是 PV 矩阵乘取操作数的必经之路，不是可以绕开的写法。

**对 `hca_unified_attention` 的六次重写、五种不同的编译器拒绝
（gather_row 的地址空间对 ×2、tile.assemble 同错、Mat 片不能跨迭代、
non-mat tmov 形状不匹配 ×2），已经足以判定：
这个函数在 PTOAS 下的问题不是某一处写法，而是它整体依赖 Mat 的方式。**

## 84. ⚠ 修正 §83 的归因：旧 allocator **已经能**细分已释放区域（2026-09-29）

§83 把 `enable_pypto_l0c_double_buffer` 与 `ak128` 的
`Right buffer usage (131072 bytes) exceeds platform limit (65536 bytes)`
都归因为"旧 PYPTO allocator 顺序堆叠复用类、从不细分已释放区域（issue #1908）"。
**这个归因是错的，它来自文档而不是实测。**

### 84.1 文档描述的是历史状态

`pypto/docs/zh/dev/passes/18-auto_tile_matmul_l0.md:77` 那段确实这么写着，
而且 `auto_tile_matmul_l0_pass.cpp:3281` 至今有一道 #1908 guard：PYPTO planner
下强制把 Mat-scratch 链式 matmul 的生产者改成 output-stationary。

但**细分能力后来已经在 `MemoryReuse` 里补上了**，而且用的正是"先不细分、
溢出了才重试"的策略（`memory_reuse_pass.cpp:3640` 起）：

```cpp
std::vector<PackedBuffer> buffers = pack(/*allow_subranges=*/false);
auto retry_with_subranges = [&]() {
  if (!subrange_space_safe || cap == 0 || footprint(buffers) <= cap) return;
  auto subdivided = pack(/*allow_subranges=*/true);
  if (footprint(subdivided) < footprint(buffers)) buffers = std::move(subdivided);
};
```

`LifetimeAnalysisResult` 里已有 `subrange_unsafe_groups`，
`whole_root_subrange_eligible` 也已按 root 的静态 offset 逐 group 判定。

### 84.2 实测：细分确实在工作

在 `pypto` 的 `fix/1908-legacy-allocator-subdivide` 分支上给
`retry_with_subranges` 加了一条环境变量门控的诊断（`PYPTO_TRACE_1908=1`），
用 `test_chained_matmul_mat_scratch_issue_1908_dsa_rp_dbc` 的形状
（m=128, k=128, nmid=512, n=64）复现：

| planner / 开关 | space | 不细分的 footprint | 细分后 | 结果 |
| --- | --- | --- | --- | --- |
| PYPTO, dbC=on | Right | 98304 | **65536** | ACCEPTED，编译通过 |
| PYPTO（#1908 guard 关）, dbC=off | Right | 81920 | **65536** | ACCEPTED，编译通过 |

四种组合（PYPTO/DSA_RP × dbC 开关）全部编译通过，一个都没复现溢出。
把 guard 关掉（`PYPTO_DISABLE_1908_GUARD=1`，同样只用于取证）后也全部通过。

### 84.3 结论与后续

**HCA 的 `Right 131072 > 65536` 另有原因，不是 allocator 不会细分。**
三道门里最可能挡住的是 `subrange_space_safe`（该 space 内存在
dynamic-offset 成员）或逐 group 的 `whole_root_subrange_eligible`；
也可能是细分后仍然装不下（`stage=2` 实际要 4 片，见第 2485 行的记录）。
要定位必须在 HCA 上跑同一条诊断——本机没有完整 vllm 运行时，只能在设备任务里取。

教训：**用文档里的限制去解释一个报错之前，先在真实路径上跑一遍诊断。**
第 33、37.5、40.2 节已经三次因为"把指标读成收益"出错，这次是同类问题的另一种形态
——把文档读成现状。

## 85. HCA 权重的 NZ 覆盖盘点：三张 matmul B 操作数还在 ND（2026-09-29）

用户问"当前 HCA 用到的所有权重都是 NZ 的么，有没有遗漏的"。逐个盘 
`_decode_hca_tp1_layer` 的签名，分三档。

### 85.1 已走 NZ（4 张，零拷贝借用 Native 存储）

经 `native_adapter.root_weight()`，布局由 `nz_mode` 的三个常量决定：

| 权重 | 形状 | dtype | 何时是 NZ |
| --- | --- | --- | --- |
| `wq_a` | [1024, 4096] | BF16 | `VLLM_ASCEND_ENABLE_NZ >= 2` |
| `wq_b` | [1024, 32768] | INT8 | `>= 1`（**默认档就是 NZ**）|
| `wo_a` | [8, 4096, 1024] | BF16 | `>= 2` **且** 当前 CANN 支持三维 BF16 cast |
| `wo_b` | [8192, 4096] | INT8 | `>= 1` |

### 85.2 ✗ 仍是 ND 的 cube B 操作数（3 张）

经 `native_adapter.weight()`，被 `npu_format_cast(..., ND)` 强制转成 ND：

| 权重 | 形状 | 消费者 | 怎么用 |
| --- | --- | --- | --- |
| `wkv` | [512,4096] 加载期转置成 [4096,512] | `qkv_proj_rope.py` 的 `kv_proj_matmul` | 正常 B 操作数 |
| `cmp_wkv` | [512, 4096] | `decode_compressor_ratio128.py` 的 `hca_kv_score_proj` | `b_trans=True` |
| `cmp_wgate` | [512, 4096] | 同上 | `b_trans=True` |

`weight()` 里的注释把它们称作"非目标权重"，但判据应该是**它是不是 matmul 的
B 操作数**——是，就该考虑 NZ，否则 cube 每次都要走片上 ND→NZ 转换
（`TLoadGm2L1Nd2Nz`）。注意这条路径**没有** `_recast` 那样的告警
（只有四张目标权重有），所以缺口容易长期看不见。

### 85.3 不需要 NZ 的（NZ 只对 cube 操作数有意义）

1D 的 norm 权重（`attn_norm_w`、`gamma_cq`、`gamma_ckv`、`cmp_norm_w`）、
量化 scale（`wq_b_scale`、`wo_b_scale`）、`attn_sink`；
mHC 门控 `hc_attn_fn` / `hc_attn_scale` / `hc_attn_base` 只进
`hc_pre_norm` 的向量路径，不是 matmul 操作数；`cmp_ape` 是 FP32 位置编码。

### 85.4 实测前提：NZ 与 `b_trans=True` 可以并存

`cmp_wkv` / `cmp_wgate` 是转置使用的，所以先单独验证这个组合能不能编译
（最小 PyPTO 程序，`private` 隔离环境）：

| 形态 | 结果 |
| --- | --- |
| NZ `[HEAD_DIM, D]` + `b_trans=True`（HCA 现状的形状） | **OK** |
| NZ `[D, HEAD_DIM]` 正常 B 操作数（`wq_a` 的形状） | OK |

**所以不需要改 kernel 的数学方向**，只加布局标注即可。

### 85.5 代价：NZ 权重不能再由 Vector 核预热

`weight_warm.py` 的 `hca_warm_kv_weights` 让 8 个空闲 Vector 核按块读这三张
权重，把它们带进 L2（冷 L2 下单个 AIC 读权重只有约 24 GB/s）。改成 NZ 后
这条路走不通，PyPTO 明确拒绝：

```
ValueError: NZ layout currently supports only matmul operand loads
(target_memory=pl.Mem.Mat), got Vec. An NZ tensor is a cube weight:
load it into Mat, or annotate the tensor as pl.ND.
```

机制：`pl.NZ` 让 `BlockNzTensorViews` 把张量改写成分形 rank-5 形状并改写
`tile.load` 坐标，生成 `TLoadGm2L1Nz2nz`（GM 分形序直搬 L1），这是 cube
操作数通路；预热用的 `[64, 512]` 二维开窗在分形排布下内存不连续，Vec 通路
也没有分形→逻辑的转换单元。

`BlockNzTensorViews` 的文档留了一条出路：**覆盖全部元素的 rank-1 view 与
layout 无关**（分块重排的是索引空间而不是内存），所以整块展平后仍可按 ND 读，
文档原话是"没有这条规则，给权重标注 `pl.NZ` 就会悄悄让它失去 SDMA L2 预热"。
`reshape` 必须发生在 Orchestration 层（InCore 里张量已变成 tile）。
PyPTO 另有 `pl.prefetch.async_prefetch`（SDMA、不占 Vector 核，要求源是
flat 1D），但那需要运行时开 SDMA，先不引入——按用户裁定"能用 vector 就先继续用 vector"。

### 85.6 逐项测量的口径

按用户要求"逐个测试，有收益才接入，判据是绝对时间，测完再验精度"，
造了 `variants_nzweights_20260929/{nz_wkv,nz_cmpwkv,nz_cmpwgate,nz_all}`
四个快照，每个只把一张（或全部三张）权重切成 NZ，并把它那段 Vector 预热循环
删掉。所以**量到的是净效应**：cube 以 NZ 读 B 操作数的收益，减去该张权重
失去 L2 预热的损失。

生产侧同时做了一处通用改动（对基线零影响）：
`nz_mode.B_OPERAND_WEIGHTS` 扩展到 7 张，`root_weight_layouts()` 只读根签名里
实际存在的参数（CSA 没有 `cmp_wgate`）；`native_adapter.weight()` 新增
`layout_name`，声明为 NZ 时在加载期一次性 `npu_format_cast(..., 29)`。
基线签名不带 layout → 读出 "ND" → 行为与改前逐字节一致。

## 86. 三张 B 操作数权重改 NZ 的逐项测量（2026-09-29）

按"逐个测、判据是绝对时间、测完验精度"的要求，`variants_nzweights_20260929/`
下每个快照只切一张权重。

### 86.1 第一批：NZ 但**去掉**该权重的 Vector 预热（128K/B16 同卡 ABBA）

| 变体 | 样本 | 中位数 | 相对 base | 精度 |
| --- | --- | --- | --- | --- |
| base | 580.2 / 582.2 / 580.2（σ≈1.2） | 580.25 | — | OK |
| `nz_cmpwkv` | 554.5 / **596.2 / 597.5** | 596.25 | **+16.00 ✗** | OK |
| `nz_cmpwgate` | 570.5 / 574.0 | 572.25 | −8.00（σ 12.6~29.7，不显著） | OK |
| `nz_wkv` | — | — | 编译失败，见 86.3 | — |
| `nz_all` | — | — | 同上 | — |

`nz_cmpwkv` 的 554.5 是孤例，后两个样本一致落在 596~598。
**两张都没拿到收益。** base 的 σ 只有 1.2，而变体样本散得多——这本身就指向
失去 L2 预热：权重是否还在 L2 里，变成取决于上一个 pass 留下了什么。

### 86.2 NZ 权重为什么不能再由 Vector 核预热，以及三次修法

PyPTO 的拒绝是显式的用户级 CHECK：

```
ValueError: NZ layout currently supports only matmul operand loads
(target_memory=pl.Mem.Mat), got Vec. An NZ tensor is a cube weight:
load it into Mat, or annotate the tensor as pl.ND.
```

机制：`pl.NZ` 让 `BlockNzTensorViews` 把张量改写成分形 rank-5 形状、改写
`tile.load` 坐标，生成 `TLoadGm2L1Nz2nz`（GM 分形序直搬 L1），这是 cube
操作数通路；预热的 `[64, 512]` 二维开窗在分形排布下内存不连续，
Vec 通路也没有分形→逻辑的转换单元。

出路是 `BlockNzTensorViews` 文档里那条规则：**覆盖全部元素的 rank-1 view
与 layout 无关**（分块重排索引空间、不动内存），展平后仍按 ND 读。
展平成 `[1, HEAD_DIM*D]` 按 32768 元素分段，正好 64 段、每段 64 KiB，
与现在的 64 个 `64×512` 块**段数和字节数都相同**。三次尝试：

| 尝试 | 做法 | 结果 |
| --- | --- | --- |
| `nz_cmp_warm` | 调用点内联 `pl.reshape(cmp_wkv, [1, N])` 传进去 | ✗ `@pl.jit: missing inferred tensor metadata for parameter 'cmp_wkv'` |
| `nz_cmp_warm2` | 先落成具名变量再传 | ✗ 同一个错误 |
| `nz_cmp_warm3` | **展平搬进 `warm_kv_weights` 函数体内**，参数仍是原始 NZ 张量 | 在测 |

**`@pl.jit` 的参数元数据推断只认直接传进来的根参数，不接受派生 view。**
错误提示自己写着 "Pass the tensor directly as a function argument"。
`warm_wo_a` 一直是对的写法——它的 `pl.reshape(wo_a, ...)` 在函数体内做，
不跨函数传。展平必须放在 `pl.spmd` 之外（spmd 体内张量已变成 tile）、
函数体之内（Orchestration 上下文）。

### 86.3 `wkv` 改 NZ 被符号证明挡住

`qkv_proj_rope.py:799` 报：

```
ValueError: NZ layout requires the slice offset on shape[-1] to be non-negative,
and this one cannot be proven to be. ... Provable forms are a non-negative
constant, the SPMD block index, a loop variable whose start and step are both
non-negative, and any sum or product built from those — note that a difference
never qualifies.
```

根因在 block 分解：

```python
kv_col0   = (kbg // (KV_OK * kv_m_groups)) * KV_N_TILE        # 列，shape[-1]
kv_k_base = ((kbg // kv_m_groups) % KV_OK) * KV_SPLIT_K_TILE  # 行，shape[-2]
kv_m_group = kbg % kv_m_groups
```

`kv_m_groups = pl.min(KV_OM, pl.max(1, tile_rows // GROUP_ROWS))` 是**运行时值**。
PyPTO 只能对**常量**除数推符号（文档：对正常量的取模／整除，符号从被除数证明），
除数一旦是运行时值就证不出来。

- `nz_wkv2`：`kv_col0 = pl.max(..., 0)`，照 `_proj_a_mm_nz` 里
  `n0 = pl.max(nf, 0) * A_COL_TILE` 的手法。**✗ 同一个错误**，
  推测是 `IsProvableNonNegative` 的 `sign_budget` 撑不住
  `Max(Mul(FloorDiv(blk, Mul(const, Min(...))), const), 0)` 这个嵌套。
- `nz_wkv3`：**把列换到最内层、对编译期常量取模**，行偏移的除数也随之变成常量：

```python
kv_col0    = (kbg % (HEAD_DIM // KV_N_TILE)) * KV_N_TILE
kv_k_base  = ((kbg // (HEAD_DIM // KV_N_TILE)) % KV_OK) * KV_SPLIT_K_TILE
kv_m_group = kbg // ((HEAD_DIM // KV_N_TILE) * KV_OK)
```

总块数 `(HEAD_DIM // KV_N_TILE) * KV_OK * kv_m_groups` 与各维取值范围都不变，
只是块编号重排，所以数学等价；但 worker 到工作的映射顺序变了，
L2 局部性可能受影响，必须实测而不是假定等价。在测。

### 86.4 顺带修掉的口径不一致

`run_hca_single_layer.sh` 硬编码 `export VLLM_ASCEND_ENABLE_NZ=1`，而
`BF16_WEIGHT_NZ` 的判据是 `>= 2`——**精度校验一直跑在与性能测试
（`run_hca_compiled_case.sh` 默认 2）不同的权重布局下**，而且 BF16 权重的
NZ 标注在那里会静默失效。已改成 `${VLLM_ASCEND_ENABLE_NZ:-1}`，
默认档位不变、可从外部覆盖。

## 87. ★★★★ 资源容量下限：AIC 是七档一致的瓶颈，makespan 是它的 1.27–1.62 倍（2026-09-29）

此前只算过**依赖**下限（静态 CPM，第 63 节：七档 74–79% of makespan）。
CPM 假设核无限多，所以它回答不了"核够不够用"。本节补上**容量**下限：
把 Worker View（pid 4）里每个片段按核类型累加，AIC 总核·μs ÷ 24、
AIV 总核·μs ÷ 48，取两者较大者——这是硬件容量给出的绝对下限，
makespan 不可能低于它。

数据取自 `results/hca_swimlane_v2_20260929/download_hca_swimlanes_20260929`
（即交付给用户的那七份最新泳道）。

| 档位 | makespan | AIC 核·μs | AIV 核·μs | AIC 下限 | AIV 下限 | 资源下限 | CPM | 实际/资源下限 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128K/B4 | 449.3 | 6664 | 7917 | 277.7 | 164.9 | 277.7 | 340 | **1.62** |
| 128K/B8 | 526.2 | 8175 | 11933 | 340.6 | 248.6 | 340.6 | 400 | 1.54 |
| 128K/B16 | 638.5 | 11395 | 18242 | 474.8 | 380.0 | 474.8 | 500 | 1.34 |
| 128K/B24 | 902.5 | 16727 | 26886 | 697.0 | 560.1 | 697.0 | 710 | 1.29 |
| 8K/B16 | 536.5 | 9436 | 13793 | 393.2 | 287.4 | 393.2 | 410 | 1.36 |
| 8K/B24 | 699.5 | 12307 | 17578 | 512.8 | 366.2 | 512.8 | 540 | 1.36 |
| 8K/B32 | 766.2 | 14490 | 21221 | 603.7 | 442.1 | 603.7 | 570 | **1.27** |

两条结论：

1. **AIC 下限在七档里全部高于 AIV 下限**（AIV 只用到 AIC 的 55%~65%）。
   cube 是瓶颈资源，48 个向量核有大量余量。
2. **8K/B32 的 AIC 下限（604）已经高于 CPM 依赖下限（570）**。也就是说
   那一档卡住的不是依赖链长度，而是 cube 容量不够——这也解释了为什么
   第 68 节的 B 路线（token-split 缩短依赖链）在那些档位上一律变差：
   依赖不是那一档的约束。

### 87.1 AIC 空转的分布

按时间切片统计"AIC 忙核数"，把 `(24 − 忙核数) × 时长` 累加起来：
**8K/B32 空转 3899 核·μs（= 162.5 μs 的满载当量），128K/B16 空转 3929 核·μs
（= 163.7 μs）**，分别占 makespan 的 21% 与 26%。最大的几个窗口：

| 档位 | 窗口 (μs) | 时长 | AIC 忙 | 这段在跑什么 |
| --- | --- | --- | --- | --- |
| 8K/B32 | 701.9–766.2 | 64.3 | **0** | `hca_oproj_hc_post_1`（38 个 AIV） |
| 8K/B32 | 206.5–307.5 | 101.0 | 22 | `qproj_matmul`(20) 等，已接近满载 |
| 128K/B16 | 591.9–638.5 | 46.6 | **1** | `hca_oproj_hc_post_0`（36 个 AIV） |
| 128K/B16 | 235.0–287.8 | 52.8 | 9 | `qproj_dequant_rms_nope_rope`(21 AIV) + `hca_cmp_work_gather_0`(16) |
| 128K/B16 | 177.2–226.1 | 48.9 | 21 | `qproj_matmul`(20)，接近满载 |

**最大的一项是 oproj 尾巴：24 个 cube 核在最后 46–64 μs 里全空**，
只有 AIV 在跑输出投影后处理。这与第 40.2 节的结论并存而不冲突：
40.2 说的是 oproj **之前**那段 core-wait 等于 proj_a→quant→proj_b 链的时长、
不可回收；本节说的是 oproj **运行期间** cube 阵列整个闲着。

### 87.2 这一项在单层 bench 里无解，但在整模型里可能不是

oproj 是层的最后一个任务，它之后单层 bench 就结束了，没有别的 cube 工作
可以填进去。真实的 61 层流水里，这段可以和**下一层开头的 cube 工作**重叠
——前提是层间没有隐式栅栏。这一点无法用单层回放验证（见第 41 节关于
单层 bench 与生产口径差异的记录），需要整模型泳道才能判断。

因此把它记下来作为待验证项，而不是当作可回收的 21%~26%。
第 33、37.5、40.2、84 节已经四次栽在"把指标读成收益"上。

## 88. NZ 逐项测量的结论：三张都没有收益（2026-09-29）

### 88.1 最终数据（128K/B16，同卡 ABBA）

| 变体 | 内容 | 样本 | 中位数 | 相对 base |
| --- | --- | --- | --- | --- |
| base | — | 5 | 586.25 | — |
| `nz_wkv3` | `wkv` NZ + block 分解重排，无预热 | 4 | 601.00 | **+14.75 ✗** |
| `nz_cmpwkv` | `cmp_wkv` NZ，无预热 | 3 | 596.25 | **+16.00 ✗** |
| `nz_cmpwgate` | `cmp_wgate` NZ，无预热 | 2 | 572.25 | −8.00（样本不足，见 88.2） |

精度检查（缓冲越界 + metadata）全部 OK。

**按"有收益才接入"的判据，三张都不接入。** 生产签名保持不带 layout 标注，
只保留第 85.6 节那个通用入口机制（布局由根签名决定），
将来条件具备时加一行标注即可打开。

### 88.2 ⚠ 样本数纪律

第 68.1 节定过：**同卡 ABBA 2 样本不足以定符号，必须 ≥6 样本**
（`qproj_n128` 当初的 −10.25 加到 6 样本后翻成 +8.75~+12.12）。
按这条复核本节：`nz_wkv3` 4 样本、`nz_cmpwkv` 3 样本，两者同向且量级接近
（+14.75 / +16.00），方向可信；**`nz_cmpwgate` 只有 2 样本，而且是唯一的负值**
——正是当初那种假信号的形态，已单独补到 6 样本复测。

### 88.3 归因实验：预热到底值多少

所有 NZ 变体都同时失去了该权重的 Vector 预热，所以退化无法归因。
补两个只删预热、**不改布局**的快照（`nowarm_wkv` / `nowarm_cmp`）给预热定价：

- 若 nowarm 退化幅度与 NZ 变体相当 → NZ 本身大致中性，损失全在预热，
  那么"NZ + 保留预热"值得继续（需要 PyPTO 支持 1D 的 Vec 搬运）；
- 若 nowarm 基本不变 → **NZ 本身就是有害的**，这条路彻底否决。

### 88.4 ✗ 一个被证伪的机制假设

曾假设 NZ 变差是因为切片碎片化：BF16 的 NZ fractal 是 16×16，
`[512, 4096]` 分块成 `[1, 256, 32, 16, 16]`（列块在最外层），
`cmp_wkv[col:col+64, 0:512]` 在列块方向取满但每块只取 4 个 fractal 行、
跳过 28 个，搬运被切成 32 段；同一窗口在 ND 下每行 1 KiB 连续。

**对照 `wq_a` 后证伪**：`wq_a`[1024, 4096] 切 `[128 行, 256 列]`（列占 1/16），
`cmp_wkv`[512, 4096] 切 `[64 行, 512 列]`（列占 1/8）——两者行方向都是 1/8，
而 `cmp_wkv` 的列方向取得**更满**。按该假设它应当更有利，与实测相反。
`wq_a` 在生产 nz2 口径下一直是 NZ 且没问题，所以"转置形态 + N 维只取一片"
本身不是原因。真正的机制待查，不猜。

### 88.5 归因结果：NZ 本身有害，而 compressor 的预热是净损失

`nowarm_*` 只删预热、不改布局，128K/B16 同卡 ABBA 各 4 样本：

| 变体 | 中位数 | 相对 base |
| --- | --- | --- |
| base | 594.38 | — |
| `nowarm_wkv` | 600.50 | **+6.12** |
| `nowarm_cmp` | 589.50 | **−4.88** |

把它代回 NZ 变体，退化终于可以归因：

| 候选 | 实测总效应 | 其中删预热 | **NZ 自身** |
| --- | --- | --- | --- |
| `cmp_wkv` → NZ | +16.00 | −4.88 | **约 +21** |
| `wkv` → NZ + block 重排 | +14.75 | +6.12 | 约 +8.6 |

**结论：NZ 布局在这几张权重的形状上本身就是有害的**，不是"失去预热"造成的
假象。所以第 86.2 节那条"保留预热"的路线即使打通（需要 PyPTO 支持 1D 的
Vec 搬运）也救不回来——不必再攻。

### 88.6 ★ 意外收获：`cmp_wkv` / `cmp_wgate` 的 Vector 预热是净损失

`nowarm_cmp` 是**负的**（−4.88 μs）：删掉这两张权重的 Vector 核预热反而更快。
这份预热在泳道里是 `hca_warm_kv_weights`，占 8 个 AIV 核、590.9 核·μs
（约 74 μs 墙钟），而带来的 L2 收益不抵这个成本。

这是一个**与 NZ 无关的独立候选**，也是这一轮唯一测到的正向效应。
注意 `nowarm_wkv` 是 +6.12——**`wkv` 的预热是有价值的，不能一起删**，
两张 compressor 权重与 `wkv` 要分开处理。

已按第 68.1 节的纪律补到 8 样本复测（`abba8_nowarmcmp_*`）；
base 在那张卡上的极差达到 33.7 μs，更需要样本量。

### 88.7 ⚠ 撤回 88.5 与 88.6 的结论：4 样本在这批卡上仍然不够

88.5／88.6 用的是 4 样本。补样本后两条结论都动摇：

| 候选 | 4 样本时 | 补测后 | 结论 |
| --- | --- | --- | --- |
| `nowarm_cmp` | −4.88 | 早期 pass **+30**（base 591.25 vs 632.5/612.5） | ✗ 88.6 的"预热是净损失"**不成立** |
| `nz_cmpwgate` | −8.00（2 样本） | **−10.50（6 样本，三轮同向）** | ✓ 站得住 |
| `nz_cmpwkv` | +16.00（3 样本） | 补测中 | 待定 |

那张卡的极差达到 **58 μs**（632.5 vs 574.5）。第 68.1 节定的"≥6 样本"是下限
而不是保证——**在漂移大的卡上，6 样本也只能分辨 10 μs 量级的效应**。

因此 88.5 那张归因表（"NZ 自身约 +21 / +8.6"）依赖 `nowarm_*` 的 −4.88／+6.12，
这两个数现在都不可靠，**整张表作废**，不要引用。
88.6 的"compressor 预热是净损失"同样撤回。

目前唯一通过样本纪律的结论是 `nz_cmpwgate` −10.50 μs（6 样本、三轮独立同向）。
而 `cmp_wgate` 与 `cmp_wkv` 形状相同、同一 kernel、同样 `b_trans=True`，
一个 −10.50 一个 +16.00 在机制上讲不通，所以在 `nz_cmpwkv` 的 6 样本出来之前
不对 NZ 是否有益下总结论。

### 88.8 6 样本定稿：`cmp_wgate` 有收益、`cmp_wkv` 有害，预热本身很值钱

| 变体 | 样本 | 中位数 | 相对 base | 独立复现 |
| --- | --- | --- | --- | --- |
| `nz_cmpwgate` | 6 | 571.62 | **−10.50** | 三轮同向（−8.0 / −13 / −10.5） |
| `nz_cmpwkv` | 6 | 601.00 | **+20.25** | 两轮同向（+16.0 / +20.25） |
| `nowarm_cmp` | 6 | 609.38 | **+25.75** | — |

`nowarm_cmp` 把 compressor 两张权重的 Vector 预热定价在 **+25.75 μs**
（单张约 +12.9），所以它**很值钱**，88.6 那条"净损失"完全说反了。
把它扣掉，NZ 自身的效应是：

| 权重 | NZ 自身（估） |
| --- | --- |
| `cmp_wgate` | 约 **−23** |
| `cmp_wkv` | 约 +7 |

⚠ 这两个估值是跨卡相减（各自的 Δ 都是同卡内测的，但两个 Δ 来自不同任务／不同卡），
依赖"效应可叠加且与卡无关"这个假设，不如 Δ 本身硬。

### 88.9 为什么两张形状相同的权重表现相反（假设）

`hca_kv_score_proj` 里两个 matmul **顺序执行、共享 A 操作数**：

```python
kv_acc    = pl.matmul(first_tile, wkv[...],   b_trans=True)   # 第一个，承担 A 的 L1 加载
score_acc = pl.matmul(first_tile, wgate[...], b_trans=True)   # 第二个，复用已在 L1 的 A
```

假设：**第一个 matmul 的 B 加载路径变化会连带扰动 A 的加载调度与 L1 分配**，
第二个则不会。这能解释 `wkv`（第一个）改 NZ 变差而 `wgate`（第二个）改 NZ 变好。
可验证方向是对比两个变体的 L1 footprint 与 `PH-MR-001` 提示，尚未做。

### 88.10 ND 别名：让 NZ 与 Vector 预热共存

要拿到 `cmp_wgate` 那约 −23 μs，必须在改 NZ 的同时**保留**预热，而 PyPTO
堵掉了所有展平读法（第 86.2 节）。绕法在于**预热只需要把字节拉进 L2，
它从不消费数值**（只写哨兵防 DCE）：

给同一块存储再绑一个**不带 layout 标注**的参数，matmul 读 NZ 的那个、
预热读 ND 别名，预热代码一行不改：

```python
# 根签名
cmp_wgate:      pl.Tensor[[HEAD_DIM, D], pl.BF16, BF16_WEIGHT_LAYOUT],
cmp_wgate_warm: pl.Tensor[[HEAD_DIM, D], pl.BF16],   # ND 别名，只供预热
# native_adapter
weights["cmp_wgate_warm"] = weights["cmp_wgate"]      # 同一个张量对象，零额外显存
```

按 ND 解释 NZ 字节会得到错的值，在这里无害——没有任何计算消费它。
变体 `nz_cmpwgate_alias`，三方同卡对照（base / `nz_cmpwgate` / 别名版）在测。
若方案成立，别名版应从 −10.50 扩大到约 −23。

### 88.11 ✗ ND 别名被绑定期格式检查否决

```
ValueError: Parameter 'cmp_wgate_warm' requires base format NCHW (0) or ND (2)
```

PyPTO 校验每个参数的**实际存储格式**必须匹配签名声明的 layout，
不允许"同一块 NZ 存储再按 ND 绑一次"。这道检查是对的（防止静默读错数据），
但也堵死了零拷贝别名。要真给预热一份 ND 拷贝，每层多 4 MiB、61 层约 244 MiB，
代价不可接受。

至此"NZ 同时保留 Vector 预热"的**五条路全部走不通**：
2D 窗口（布局检查）、内联 reshape 传参（元数据推断）、具名变量传参（同上）、
rank-1 展平（codegen 无 1D GM→UB 搬运）、ND 别名（绑定期格式检查）。

### 88.12 最终判定：三张权重都不接入

`nz_cmpwgate` 第四轮（另一张卡，4 样本）测出 **+2.75**，而前三轮是
−8.00 / −13 / −10.50。四轮三负一正，效应在 −10.5 ~ +2.75 之间摇摆。

按本轮确立的分辨力标准——这批卡的 base 极差可达 58 μs，6 样本只够分辨
10 μs 量级——**−10.50 恰好落在分辨力边缘，不足以支撑接入**。

| 权重 → NZ | 结论 |
| --- | --- |
| `cmp_wgate` | 四轮符号不稳（−8.0 / −13 / −10.5 / +2.75），**不接入** |
| `cmp_wkv` | +20.25（6 样本，两轮同向），**否决** |
| `wkv` | +14.75（4 样本），**否决** |

生产代码保持基线：三张权重的签名都不带 layout 标注。
保留的只有第 85.6 节那个通用入口机制（`B_OPERAND_WEIGHTS` +
`weight(..., layout_name=)`），它对基线逐字节零影响，将来若 PyPTO 放开
NZ 张量的非 Mat 加载（预热就能保留），加一行标注即可重测。

### 88.13 这一轮留下的可复用结论

即使候选全否，有三条是后续要用的：

1. **compressor 两张权重的 Vector 预热值 +25.75 μs**（`nowarm_cmp`，6 样本）。
   这是第一次给 `hca_warm_kv_weights` 定量，说明它不是可砍的开销。
   而 `nowarm_wkv` 是 +6.12——`wkv` 的预热也有价值但小得多。
2. **NZ 与 Vector 预热在当前 PyPTO 下互斥**，五条绕法全部受阻（88.11）。
   这是给 PyPTO 提需求的具体依据：放开 NZ 张量的整块 1D Vec 读取
   （`BlockNzTensorViews` 的文档已按这条规则写了，缺的是 codegen 的 1D GM→UB）。
3. **这批卡的噪声水平**：base 单卡极差可达 58 μs、σ 约 25。第 68.1 节的
   "≥6 样本"是下限而非保证；10 μs 以下的效应在当前测量条件下不可判定。

## 89. 核间流水轴：从泳道定位到 AIC 空转的责任链，两个候选均否（2026-09-29）

### 89.1 责任链定位

128K/B16 泳道，AIC 在 **235.0→280.1 完全空转 45 μs**（`qproj_matmul` 结束到
`hca_unified_attention_aic` 开始）。那段窗口里的任务：

| 任务 | 窗口内核·μs | 起→止 | 核型 |
| --- | --- | --- | --- |
| `qproj_dequant_rms_nope_rope` | 1098.3 | 246.6→314.1 | AIV |
| `hca_cmp_work_gather_0` | 854.1 | 238.2→289.0 | AIV |
| `hca_unified_attention_aic` | 27.4 | **280.1**→478.6 | AIC |

**attention 等的不是 dequant** —— 它在 280.1 就开工，而 dequant 到 314.1 才结束，
两者重叠 34 μs，说明 `allow_early_resolve` 是有效的。attention 等的是
`hca_cmp_work_gather`（238.2→289.0）。责任链是
`state_commit → cmp_work_gather → attention`。

### 89.2 gather 的硬缺口

`cmp_gather_blocks` 在长档下被 `CMP_GATHER_WORK_TILE=2` 压成 **32 块**
（`cmp_gather_count = request_count × cmp_work_count = 64`），
每个 worker 顺序做 2 个 gather 单元。

- 占用：32 核 × 50.8 μs = **1625 核·μs**
- 实际工作：**854 核·μs**
- **缺口一倍，每个 worker 约 24 μs 在等**

### 89.3 ✗ `gather_full`：blocks 32→64 用满 48 核

去掉那个压缩（`cmp_gather_blocks = cmp_gather_count`），每 worker 只做 1 个单元。
6 样本同卡 ABBA：base 567.62，`gather_full` 586.62 → **+19.00 μs**。

**假设错了。** 64 块在 48 个 AIV 核上是 1.33 波：第一波 48 个、第二波只有 16 个，
而第二波必须等第一波整体完成（gang 调度），于是第二波浪费 32 个核。
32 块一波到底、没有波间等待，反而更优。**现有的 32 块打包是对的**，
这是第 32 节"索要满核有害"的又一个实例。

推论：那一倍缺口**不在并行度上**，而是在等待。

### 89.4 ✗ `gather_pipe`：内层循环加软流水

循环体每页是一条串行链：`pl.read(cmp_block_table)` 标量读 GM（高延迟）→
两层分支 → `pl.gather_row` 的 32 KiB DMA。4 页 × 2 单元 = 8 条串行链，
这是 89.2 那 24 μs 等待的唯一来源。把内层
`pl.range(CMP_PAGES_PER_WORK)` 改成 `pl.pipeline(..., stage=2)`，
让下一页的标量读与当前页的 DMA 重叠。

6 样本同卡 ABBA：base 570.50，`gather_pipe` 580.75 → **+10.25 μs**。

`pl.pipeline` 要额外缓冲与 stage 间同步，在只有 4 次迭代的短循环上
开销盖过了重叠收益。

### 89.5 本轮汇总：五个候选全否，1.117 未动

| 轴 | 候选 | 结果 |
| --- | --- | --- |
| 核内／布局 | `wkv` → NZ | +14.75 ✗ |
| 核内／布局 | `cmp_wkv` → NZ | +20.25 ✗ |
| 核内／布局 | `cmp_wgate` → NZ | 四轮符号不稳 ✗ |
| 核间流水 | `gather_full` | +19.00 ✗ |
| 核间流水 | `gather_pipe` | +10.25 ✗ |

七档加权仍是 **1.117**，距 0.80 差 0.317。

值得注意的是这一轮的定位方法比上一轮硬：候选不是靠猜，而是先从泳道算出
**定量缺口**（gather 1625 vs 854 核·μs、AIC 空转 162 核·μs 满载当量）再找原因。
缺口是真的，但两次尝试都表明它**不是可回收的预算**——既不能靠提并行度拿
（gang 调度的波次代价），也不能靠软流水拿（短循环的同步开销）。
这与第 33、37.5、40.2、84、88.7 节是同一条教训的第六次：
**泳道上算得出的差额不等于可回收的收益。**

## 90. ★★★★ 用 pipe 数据推翻 §58 的带宽假设：AIC 是 scalar-bound，不是搬运受限（2026-09-29）

§58.2 建了一个"attention 两条 lane 都 GM 带宽受限"的模型，并自己标注
**⚠ 未经验证**，理由是 `cmp_work_kv` 只有约 18.9 MiB、装得进 L2，往返可能根本不走
HBM；并说要验证需要 pipe 级数据，而"PTO 整个半边只有一个 kernel，
torch_npu profiler 也拆不开"。

**profiler 拆得开。** `torch_npu.profiler.AiCMetrics` 的 `PipeUtilization`
会给每个 kernel 补出 `aic_mac/scalar/mte1/mte2/fixpipe` 的 time 与 ratio 列。
`results/hca_optimization_20260928/incore_fd_v24_h131072_b16` 里就有：

| kernel | 墙钟 | mac | scalar | mte1 | mte2 | fixpipe |
| --- | --- | --- | --- | --- | --- | --- |
| PTO `aicore_kernel_mode_0` | 663.2 | **14.9%** | 59.2% | 17.2% | **29.9%** | 12.0% |
| Native `SparseAttnSharedkv` | 186.1 | 26.7% | 59.4% | 33.0% | **91.4%** | 90.8% |

### 90.1 §58 的带宽假设被推翻

**PTO 的 mte2 只有 29.9%——AIC 侧不是搬运受限。** 反倒是 Native 的 mte2 高达
91.4%（它才是贴着搬运上限跑的）。所以"AIC 的 KV 被重读 6 次 → 读了 124 MiB
→ 95 μs 搬运"这条推算不成立：唯一 KV 约 21 MiB、暂存张量约 18.9 MiB，
都在 192 MiB 的 L2 里，重读绝大部分命中 L2。

§58.3 的根因 1（KV 重读 6 次）因此**不是**当前的主要损失来源，
不值得为它做"按请求分工"那种结构性重排——那会把小 batch 的并行度打掉
（B4 只有 4 个请求，24 个 AIC 核里要空 20 个）。

### 90.2 真正的差别：mac / scalar 比

两侧的 scalar 占比几乎一样（59.2% vs 59.4%），但：

| | mac / scalar |
| --- | --- |
| Native | 26.7 / 59.4 = **0.45** |
| PTO | 14.9 / 59.2 = **0.25** |

**PTO 每单位标量工作只做到 Native 一半的 MAC。** AIC 核被占着，
但 cube 在等标量算完地址、判完边界、对完 flag。这解释了为什么
"AIC 容量下限"（按任务占用算）远高于 MAC 时间：核忙 ≠ cube 忙。

也解释了这一轮与第 58 节六轮候选为什么方向性地全否——**调块数、tiling、
预取、软流水动的都是搬运与并行度，而瓶颈在标量指令数上**。

### 90.3 顺带纠正 §87 的读法

§87 算的"AIC 容量下限"是**任务占用时间 / 24**，它包含了 scalar 等待，
所以那个 1.27~1.62 倍的差距里有一部分并不是"调度没打满"，
而是"核占着但 cube 空转"。两者要分开：
前者归核间流水，后者归核内（降标量开销）。

### 90.4 下一步应该攻什么

降 AIC 的标量指令数，而不是搬运。可查的方向（均未验证）：

- attention tick 循环每 tick 的标量工作量（页表读、索引推算、边界判断）；
  36 个 tick × 24 块，标量是按 tick 线性增长的。
- `ATTN_K_TILE` 增大可摊薄每 tick 标量，但 128→256 会让
  `kv_l1`（3 slot × K × 512 × 2B）从 384 KiB 涨到 768 KiB，超 512 KiB L1——
  **这条路被 L1 容量堵死**，除非同时把 slot 数从 3 降到 1（那会丢掉 QK 的软流水）。
- 用 `PYPTO_LOG_LEVEL=info` 的 `perf_hints.log` 找标量热点，
  以及确认 `pl.read` 这类 GM 标量读在 tick 循环里出现了多少次。

### 90.5 量出标量热点：每 tick 6 次 `% 3`，而 3 不是 2 的幂

按 90.4 去量 `hca_unified_attention` 的 tick 循环体（`decode_sparse_attn_hca.py`
第 1191–1324 行，134 行）：

| 构造 | 次数/​tick | 说明 |
| --- | --- | --- |
| `pl.read(` | **0** | 循环内没有 GM 标量读，这一项是干净的 |
| `//` | 17 | **绝大多数是编译期常量**（`H // 2`、`HEAD_DIM // 2`、`HEAD_DIM // PV_N_TILE`、`H // 2 // H_TILE`），会被折叠 |
| `%` | **6** | **全部是运行时取模**，见下 |
| `pl.cast(` | 7 | — |
| `pl.gather_row` / `pl.matmul` | 2 / 1 | 实际计算 |

六处运行时取模，除数都是 `QK_TRANSFER_SLOTS`：

```python
l1_row     = (tick % QK_TRANSFER_SLOTS) * ATTN_K_TILE
row        = (worker * QK_TRANSFER_SLOTS + tick % QK_TRANSFER_SLOTS) * H
pv_row     = (worker * QK_TRANSFER_SLOTS + pv_work % QK_TRANSFER_SLOTS) * H
pv_l1_row  = (pv_work % QK_TRANSFER_SLOTS) * ATTN_K_TILE
vec_row    = (worker * QK_TRANSFER_SLOTS + vec_tick % QK_TRANSFER_SLOTS) * H + head0
out_row    = (worker * QK_TRANSFER_SLOTS + out_work % QK_TRANSFER_SLOTS) * H + head0
```

`QK_TRANSFER_SLOTS = QK_PRE_LAUNCH + 1 = 3`——**不是 2 的幂**，所以编译器不能降成
位与，只能出真除法或魔数乘法＋移位。36 tick × 6 次 = 每 worker 216 次除法类运算，
24 个 worker 合计约 5200 次。这与 90.2 的 mac/scalar 比（PTO 0.25 vs Native 0.45）
方向一致。

### 90.6 ✗ 把 `QK_TRANSFER_SLOTS` 改成 4 被 L1 容量堵死

4 是 2 的幂，取模能降成位与。但 `kv_l1` 是
`pl.create_tile([QK_TRANSFER_SLOTS * ATTN_K_TILE, HEAD_DIM], BF16, Mat)`：

| SLOTS | kv_l1 | 结论 |
| --- | --- | --- |
| 3（现状） | 3 × 128 × 512 × 2B = **384 KiB** | 装得下 |
| 4 | 4 × 128 × 512 × 2B = **512 KiB** | **正好占满整个 Mat 空间**，query 与权重无处可放 |

和 90.4 里 `ATTN_K_TILE` 128→256 是同一道墙。**不要再往这个方向调常量。**

### 90.7 下一步：用循环携带的 slot 计数器消掉取模（未实现）

除数改不动，但取模本身可以不用除法算——`slot` 每 tick 只 +1、到 `SLOTS` 归零，
用 `pl.range` 的 `init_values` 携带它，归零用比较＋条件减（无除法、无 `if`，
避免 `@pl.jit` trace 两个分支）：

```python
for tick, (slot,) in pl.range(work_count + QK_PRE_LAUNCH, init_values=[0]):
    l1_row = slot * ATTN_K_TILE          # 原来是 (tick % SLOTS) * ATTN_K_TILE
    ...
    nxt = slot + 1
    pl.yield_(nxt - QK_TRANSFER_SLOTS * pl.cast(nxt >= QK_TRANSFER_SLOTS, pl.INDEX))
```

六处取模分布在 AIC lane（`tick`／`pv_work`）与 AIV lane（`vec_tick`／`out_work`）
的不同循环里，要各自携带一个计数器。`slot` 是标量而非 Mat tile，
不触犯"Mat-resident tile 不能做 loop-carried iter_arg"那条限制。

⚠ 尚未实现与验证。预期收益量级：若 216 次除法/worker 每次按 10~20 周期估，
约 2~4 μs/worker——**这个量级低于本批卡的分辨力（10 μs）**，
所以要么等它与其它标量削减叠加后一起量，要么改用 incore 模拟器（`incore-profiling`
技能）在周期级验证，而不是拿同卡 ABBA 去测。

### 90.8 ✗ 先否掉 90.7：取模只占标量的 0.5%

在实现之前先做数量级校验（这正是本轮教训的应用）：

| 量 | 值 |
| --- | --- |
| AIC 平均单核的 scalar 时间 | **387.1 μs**（占 654 μs 墙钟的 59.2%） |
| 216 次除法/worker × 约 15 周期 ÷ 1.85 GHz | 约 **1.8 μs** |
| 占比 | **0.5%** |

**90.7 的 slot 计数器改法不值得实现。** 顺带确认 `aic_total_cycles`
（29 036 926）÷ `aicore_time`（654.0 μs）= 44.40 GHz ≈ 24 × 1.85 GHz，
即 cycles 是 24 核总和而 `aicore_time` 是墙钟，两者与 ratio 同口径，数据可信。
`aic_icache_miss_rate` 只有 0.001%，也排除了取指问题。

### 90.9 标量没有单一热点，而减少它的唯一出路被 L1 堵死

387.1 μs 的标量摊在「134 行循环体 × 144 次执行」（36 tick × 每 worker 4 token）上，
约 2.69 μs／次、4840 周期。**没有哪一处占大头**，所以不存在"改一处就见效"的
标量优化；唯一能整体削减的办法是**减少循环次数**，那要求增大 tile，
而增大 tile 撞 L1（90.6：`kv_l1` 已 384 KiB，再大就占满 512 KiB）。

于是形成一条闭合的否决链：

```
scalar 是瓶颈（59.2%，最大 pipe）
  → 但无单一热点（均摊在 134 行 × 144 次）
  → 只能靠减少循环次数
  → 需要更大的 tile
  → 被 512 KiB L1 容量堵死
```

### 90.10 ★ 与 Native 的真正差距：L1 用量与流水深度

同样的 scalar 占比（PTO 59.2% / Native 59.4%），Native 的 mac 却是我们的 1.8 倍
（26.7% vs 14.9%）——**它在同样的标量开销下做了更多 MAC**。第 63.3 节记的
`ops-transformer` 配方给出了原因：

| | L1 用量 | 流水深度 |
| --- | --- | --- |
| Native `SparseAttnSharedkv` | `bufQPL1` 256 KiB + `bufKVL1` 192 KiB = **448 / 512 KiB** | **7**（4 + 3） |
| PTO `hca_unified_attention` | `kv_l1` **384 / 512 KiB** | **3**（`QK_TRANSFER_SLOTS`） |

Native 把 L1 用到 87.5% 并做到 7 级流水；我们用 75% 只做到 3 级。
每 tick 的标量开销被摊到更多的 MAC 上，mac/scalar 就从 0.25 升到 0.45。

**所以 0.80 需要的是 attention 的 L1 布局与 tile 尺寸重新设计
（更大 tile、更深流水、Q 与 KV 分开分配），不是继续调参。**
本轮 5 个候选、第 58 节 6 个候选、以及 90.6～90.8 的三次否决，
都是在"当前分解 + 调参"这个空间里穷举，而这个空间已经证明是封闭的。

⚠ 这是结构性重写，影响 attention 的正确性与所有七档，
不应在没有完整回归（逐 bit + 七档）的前提下动手。

### 90.11 ⚠ 修正 90.10：L1 利用率我们比 Native 还高，差距在每 tick 的标量量

90.10 说"Native 用 448/512 KiB（87.5%），我们只用 384（75%）"——**这个对比是错的**，
我只算了 `kv_l1`，漏掉了同样驻留 Mat 的其它 tile。实际清点长路径 attention：

| Mat 驻留 tile | 尺寸 | 占用 |
| --- | --- | --- |
| `query` = `pl.load(q_flat, [token*H, 0], [H, HEAD_DIM], Mat)` | 64 × 512 × 2B | **64 KiB** |
| `kv_l1` = `pl.create_tile([SLOTS*ATTN_K_TILE, HEAD_DIM], Mat)` | 3 × 128 × 512 × 2B | **384 KiB** |
| `probability` = `pl.load(probs, [pv_row, 0], [H, ATTN_K_TILE], Mat)` | 64 × 128 × 2B | **16 KiB** |
| 合计 | | **464 / 512 KiB = 90.6%** |

**我们的 L1 利用率（90.6%）高于 Native（87.5%）。** 所以"提高 L1 利用率"
这条路不存在，90.10 那句"0.80 需要重新设计 L1 布局"的依据不成立。

### 90.12 真正的差别：块大小与每 tick 的标量量

拆开 Native 的配方（第 63.3 节，`L1_BLOCK_SIZE = 64 × 512 × sizeof(Q_T) = 64 KiB`）：

| | KV 块 | KV 级数 | KV 占用 | Q/P 占用 |
| --- | --- | --- | --- | --- |
| Native | **64 行** | 3（`bufKVL1`） | 192 KiB | 256 KiB（`bufQPL1`，4 级） |
| PTO | **128 行** | 3（`QK_TRANSFER_SLOTS`） | 384 KiB | 64 + 16 = 80 KiB |

**Native 的 KV 块只有我们的一半**，同样 3 级却只花 192 KiB，省下的给了 Q/P 的 4 级。
它的"7 级"是 4（QP）+ 3（KV），不是 KV 本身 7 级——90.10 把两者混为一谈了。

那我们把 `ATTN_K_TILE` 128→64 是否就能对齐？L1 会省下 192 KiB，但
**tick 数从 36 翻到 72，每 tick 的标量开销照样翻倍**——而 90.9 已经量出
标量是均摊的、无热点。第 58 节记的 `kt64` 实测更差，与此一致。

于是差距落在一个更难办的地方：**Native 每 tick 的标量比我们少**。
两侧 scalar 占比相同（59.4% / 59.2%）而 Native 的 mac 高 1.8 倍，
意味着在同样的标量预算下它推进了更多 MAC。它是手写 AscendC，
地址计算、边界判断、flag 交接都能人工精简；PyPTO 从 DSL 生成的这部分更冗余。

**这不是算子侧调参能消除的差距，属于 PyPTO 的代码生成质量。**
要缩小它，方向是让 PyPTO 为这个循环生成更精简的标量序列
（例如把循环不变的地址计算提到循环外、把多次派生的索引合并），
而不是在算子里换常量。

⚠ 这一节只是把差距归到了 codegen，尚未取证具体是哪些标量指令冗余。
下一步应当用 `compare-codegen` 技能 dump 出这个 kernel 的 `.pto`／汇编，
数出每 tick 的标量指令条数，再与 Native 的 AscendC 汇编对照。

## 91. ★★★★★ scalar 的真正来源：每 tick 82 个流水同步原语（2026-09-29）

90.12 把差距归到"codegen 的标量冗余"但没取证。取证方法：PTO 的编译产物里
**有生成的 AscendC 源码**，不必重新编译。位置：

```
results/hca_swimlane_v2_20260929/h131072_b16/build_output/
  _jit__decode_hca_tp1_layer_<hash>/kernels/aic/hca_unified_attention_aic.cpp
```

（同目录下 43 个 `.cpp` + 42 个 `.pto`，`kernels/aic/` 与 `kernels/aiv/` 分开。）

### 91.1 数据

`hca_unified_attention_aic.cpp`（3213 行）：

| 原语 | 全文件 | 最内层 tick 循环体内 |
| --- | --- | --- |
| `wait_flag` | 86 | — |
| `set_flag` | 83 | — |
| `pipe_barrier` | 79 | — |
| **同步合计** | **248** | **82** |
| `TASSIGN` | 198 | — |
| `TEXTRACT` | 10 | — |

**每 tick 82 个同步原语。** × 36 tick × 每 worker 4 token ≈ **11 800 次/worker**。
生成代码的实际形态（最内层）：

```cpp
uint64_t v173 = (uint64_t) v29;
TASSIGN(v172, v173);
wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
pipe_barrier(PIPE_MTE1);
uint16_t v174 = static_cast<uint16_t>(v41);
uint16_t v175 = static_cast<uint16_t>(v171);
TEXTRACT(v172, v132, v174, v175);
```

### 91.2 这解释了前面所有的否决

`wait_flag` 每次停等数十到数百周期，`pipe_barrier` 是全流水屏障。
**387 μs 的 AIC scalar 主体是同步等待，不是指令执行**——于是：

| 此前的否决 | 为什么必然否 |
| --- | --- |
| 提并行度（`gather_full`、v35a、qt8/16） | 不减少每 tick 的同步数 |
| 软流水（`gather_pipe`） | 反而**增加** stage 间同步 |
| NZ 布局（三张权重） | 只改搬运路径，同步数不变 |
| 取模改位运算（90.7） | 只占 0.5% |
| 增大 tile（90.6） | 被 L1 堵死，且不减同步密度 |

也解释了 90.2 的 mac/scalar 差（Native 0.45 / PTO 0.25）：Native 是手写
AscendC，同步点由人按数据流合并；PyPTO 为每个 tile 操作保守地插一对 flag。

### 91.3 下一步：减少同步密度

两个层次，都需要先量化"82 个里哪些是冗余的"：

1. **算子侧**：减少 tick 循环体内的 tile 操作数量（每个操作都会引入同步）。
   循环体 134 行里有 2 次 `gather_row`、2 次 `tile.extract`、1 次 `matmul`、
   7 次 `cast`——`TASSIGN` 全文件 198 次说明临时 tile 赋值很多。
2. **PyPTO 侧**：同步插入策略。`pipe_barrier` 79 次（全流水屏障）里，
   若能替换成针对具体 pipe 的 `set_flag`/`wait_flag` 对，或合并相邻屏障，
   代价会显著下降。这属于 codegen，按约定要先商量。

⚠ 尚未量化冗余比例。可行的取证：对照 `kernels/aiv/hca_unified_attention_aiv.cpp`
的同步密度，以及用 `compare-codegen` 比较不同写法下的同步数变化——
**这是第一次有了可以直接计数的指标，不必再靠 ABBA 的 10 μs 分辨力去猜。**

### 91.4 ⚠ 更正 91.1：那 82 个同步大部分属于 AIV 半边，不是 AIC 的

两个信号说明 91.1 的归因错了：

| 文件 | 行数 | `wait_flag` | `set_flag` | `pipe_barrier` |
| --- | --- | --- | --- | --- |
| `hca_unified_attention_aic.cpp` | 3213 | 86 | 83 | 79 |
| `hca_unified_attention_aiv.cpp` | 3216 | 86 | 83 | 79 |

1. **两个 lane 的计数完全相同** —— 它们是同一份 MIX kernel 源码的两个编译产物
   （`if ASCEND_IS_AIC` / `if ASCEND_IS_AIV` 结构），不是各自独立的代码。
   所以在 aic 文件里数出的同步数**不等于** AIC 实际执行的同步数。
2. **按 pipe 分类，79 个 barrier 里 64 个是 `pipe_barrier(PIPE_V)`**：

| pipe | 次数 | 归属 |
| --- | --- | --- |
| `PIPE_V` | **64** | 向量流水——**AIC 不跑向量指令**，属 AIV 半边 |
| `PIPE_FIX` | 5 | AIC |
| `PIPE_M` | 4 | AIC |
| `PIPE_MTE3` | 2 | 主要 AIV |
| `PIPE_MTE1` | 2 | AIC |
| `PIPE_ALL` | 2 | 两者 |

AIC 半边实际只有约 **15 个 barrier**（FIX 5 + M 4 + MTE1 2 + MTE3 2 + ALL 2）。

**所以"AIC 每 tick 82 个同步原语导致 387 μs scalar"这条归因不成立，撤回。**
91.2 那张"解释了所有否决"的表也失去依据——那些候选为什么否，回到未解释状态。

### 91.5 仍然成立的部分，与下一步该怎么取证

没有被推翻的：

- AIC 的 scalar 确实是最大 pipe（59.2%，平均单核 387 μs / 654 μs 墙钟，第 90 节）；
- scalar 均摊在 134 行循环体 × 144 次执行上、无单一热点（第 90.9 节）；
- 取模只占 0.5%（第 90.8 节）、L1 已用 90.6% 高于 Native（第 90.11 节）、
  mac/scalar 是 Native 0.45 / PTO 0.25（第 90.2 节）。

要把 scalar 落到具体构造上，必须**先按 `ASCEND_IS_AIC` 分支切出 AIC 半边再统计**，
而不是对整个 `.cpp` 计数。`wait_flag` 86 / `set_flag` 83 也要同样区分：
其中跨核交接（`PIPE_M`↔`PIPE_MTE1`、cross-core flag）才属 AIC 的等待。

教训：**生成的 MIX kernel 源码里两个 lane 的代码共存，按文件名当成单 lane 统计会
把另一半的同步算到自己头上。** 这是本轮第五次更正（前四次：两次小样本当效应、
一次漏算 L1 占用、一次误读 Native 流水级数），也是"用两个不同口径的数相减／
相认就下结论"这类错误的又一次，与第 41、50、84、88.7 节同类。

### 91.6 ★ 正确的口径：MIX kernel 是两个并列函数，按行切开后 AIC 半边 78 个同步

91.4 说要"按 `ASCEND_IS_AIC` 分支切"——**实际不是宏分支**。生成的
`hca_unified_attention_aic.cpp` 里是**两个并列的 `__aicore__` 函数**：

```
第  60 行  static __aicore__ void hca_unified_attention_aic(...)
第 817 行  static __aicore__ void hca_unified_attention_aiv(...)
```

两个 `.cpp`（`kernels/aic/` 与 `kernels/aiv/`）内容相同、都含这两个函数，
编译时选不同入口——这就是 91.4 看到"两 lane 计数完全相同"的原因。
正确的统计要按**函数行范围**切（AIC 为 60–816）：

| lane | 行数 | `wait_flag` | `set_flag` | `pipe_barrier` | CrossCore | 同步合计 | 密度 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **AIC** | 757 | 34 | 33 | **11** | 2 | **78** | **每 10 行 1 个** |
| AIV | 2397 | 51 | 49 | 67 | 5 | 167 | 每 14 行 1 个 |

AIC 的 11 个 barrier 全部是 `PIPE_FIX`(5) / `PIPE_M`(4) / `PIPE_MTE1`(2)——
都是 AIC 自己的 pipe，归属正确；64 个 `PIPE_V` 确实都在 AIV 半边。

**所以 91.1 的方向成立、数字口径错**：不是"整文件 82 个"也不是"每 tick 82 个"，
而是 **AIC 函数体 757 行里 78 个同步原语，密度高于 AIV**。
91.2 那张"解释了所有否决"的表因此**恢复依据**（91.4 里对它的撤回收回）：
提并行度、软流水、NZ 布局、改常量都不减少这 78 个同步。

⚠ 仍未测的一步：这 78 个里有多少落在 tick 循环体内（即随 36 tick × 4 token 放大），
有多少是函数级一次性的。这决定了可回收的量级，是下一轮的第一件事——
`/tmp` 里的切分方法可复用：`sed -n '60,816p'` 取 AIC 半边后再定位循环。

### 91.7 ★★★★★ 闭合：64 个同步落在最内层 K 循环里，×576 后与 387 μs 吻合

把 AIC 半边（`sed -n '60,816p'`）的循环结构与同步行号对齐：

| 循环（相对行） | 含义 |
| --- | --- |
| 178 `for (v112 = v110; v112 < v18; v112 += v39)` | token（步长 = `NUM_QK_CORES` 24） |
| 278 `for (v144 = v41; ...)` | tick |
| 376 `for (v171 = v41; v171 < v35; v171 += v50)` | **K 方向分块** |

78 个同步按层次分布：

| 位置 | 同步数 | 每 worker 放大 | 执行次数 |
| --- | --- | --- | --- |
| 函数序言（<178） | 6 | ×1 | 6 |
| token 内、tick 外 | 3 | ×4 | 12 |
| tick 内、最内层外 | 5 | ×36×4 | 720 |
| **最内层 K 循环内（≥376）** | **64** | **×4×36×4 = ×576** | **36 864** |

**36 864 × 约 20 周期 ÷ 1.85 GHz ≈ 400 μs**，与实测 AIC scalar **387 μs**
（第 90 节）数量级吻合。**387 μs 至此有了闭合的解释。**

### 91.8 问题的形状：最内层做 3 件事却带 64 个同步

那个最内层循环体（生成代码）只做三件事：

```cpp
Tile<TileType::Left,  bfloat16_t, 64, 128, ...>   // qk 的 A
TASSIGN(v172, v173);  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);  pipe_barrier(PIPE_MTE1);
TEXTRACT(v172, v132, v174, v175);
Tile<TileType::Right, bfloat16_t, 128, 128, ...>  // qk 的 B
TASSIGN(v176, v177);  ...
TEXTRACT(...);
// 然后 matmul
```

即 **2 次 `TEXTRACT` + 1 次 `matmul`，却插了 64 个同步原语**。
这个比例说明 PyPTO 为每个 tile 操作保守地成对插 flag，而没有跨迭代合并
（同一个 slot 在下一次迭代才被覆盖，本可以只在 slot 轮转边界同步一次）。

两个方向：

1. **算子侧**：减少最内层的 tile 操作数。`TEXTRACT` 是
   `pl.tile.extract(kv_l1, ...)` / `pl.tile.slice(pl.tile.transpose_view(...))`
   产生的——若能把 K 循环的 extract 提到循环外（一次 extract 整块、循环内只切 view），
   同步数会按比例下降。**这是下一轮第一个该试的改动。**
2. **PyPTO 侧**：同步插入的粒度。64 : 3 的比例不像是必要的数据依赖，
   更像是"每个 tile 操作一对 flag"的保守策略。属 codegen，按约定先商量。

⚠ 收益上限估算：若最内层同步从 64 降到 16（只在 slot 轮转边界同步），
36 864 → 9 216，按同样系数约省 **300 μs** 的 AIC scalar。
即便只减一半也有约 200 μs——**这是本项目迄今识别出的最大单项**，
远超此前所有候选（最好的一次落地是 −47 μs）。
但估算依赖"每个同步约 20 周期"这个未实测的系数，
**先用计数验证改动方向，再用同卡 ABBA 量绝对收益。**

### 91.9 ⚠ 更正 91.7 的分布：最内层 K 循环体只有 72 行 / 13 个同步

91.7 把"相对行 ≥ 376 的同步"全算作最内层 K 循环内，**错了**：那个循环
（`for (v171 = v41; v171 < v35; v171 += v50)`）在相对行 376 开始、**447 结束，
只有 72 行**，其后的同步属于 tick 循环体的后半段（PV matmul 与输出处理，
即 `if tick >= QK_PRE_LAUNCH` 分支）。

按真实边界重算：

| 区间（`aic_only` 相对行） | 同步数 | 每 worker 放大 | 执行次数 |
| --- | --- | --- | --- |
| 函数序言 1–177 | 6 | ×1 | 6 |
| token 内 / tick 外 178–277 | 3 | ×4 | 12 |
| tick 内 / K 外 278–375 | 5 | ×144 | 720 |
| **最内层 K 循环体 376–447** | **13** | ×4×36×4 = ×576 | **7 488** |
| **K 循环后、仍在 tick 内 448–757** | **51** | ×36×4 = ×144 | **7 344** |
| 合计 | 78 | | **≈ 15 570** |

**总执行数 15 570，不是 91.7 说的 36 864。** 反推每个同步约
387 μs × 1.85 GHz ÷ 15 570 ≈ **46 周期**（91.7 假设的 20 周期偏低）。

两点结构性修正：

1. **有两个热点，量级相当**（7 488 与 7 344），不是单一热点。
   最内层 K 循环那 13 个来自 QK matmul 的 K 分块（2 次 `TEXTRACT` + `matmul`）；
   另外 51 个在 PV 与输出处理段。**只优化 QK 的 extract 最多拿回一半。**
2. **收益上限相应下调**：若整体同步减半 → 约省 **190 μs**（91.8 估的 300 μs 偏高）。
   仍是迄今最大单项（此前最好落地 −47 μs），但不足以单独填满 0.317 的差距。

这是本轮第七次口径更正。前六次：两次小样本当效应、漏算 L1 占用、误读 Native
流水级数、把 AIV 半边同步算到 AIC、过度撤回一个正确结论。
**共同模式都是"在没确认边界／口径之前就开始算"**——
本轮真正该记住的方法论就是这一条，而不是任何单个数字。

## 92. ★★★★★ 实验证伪第 91 节：同步数降 64%，耗时反而 +12.88 μs（2026-09-29）

第 91 节的结论链是"AIC scalar 387 μs ← 15 570 次同步 × 约 46 周期"，
并据此提出"同步计数是不受卡噪声影响的确定性指标"。**做了一次改动就把它证伪了。**

### 92.1 改动：QK 从自动 K 分块改成手工 split-K

同一个 tick 循环体里两条 matmul 路径的写法本来不一致：

| 路径 | 写法 | L0 管理 |
| --- | --- | --- |
| PV | `pl.tile.move(...Left)` + `pl.tile.extract(...Right)` + `pl.tile.matmul` | 手工 |
| QK | `pl.matmul(query, key_t)` | 编译器自动 K 分块 |

那 13 个 ×576 的同步正是编译器为 QK 生成的 K 循环带来的。于是照 PV 的写法改 QK：

```python
qk = pl.create_tile([H, ATTN_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
for qk_k in pl.unroll(HEAD_DIM // QK_K_TILE):          # 编译期展开，无循环控制同步
    qk_a = pl.tile.extract(query, 0, qk_k*QK_K_TILE, [H, QK_K_TILE], Left)
    qk_b = pl.tile.extract(key_t, qk_k*QK_K_TILE, 0, [QK_K_TILE, ATTN_K_TILE], Right)
    qk = pl.tile.matmul_acc(qk, qk_a, qk_b, init_cond=(qk_k == 0))
```

### 92.2 结果：同步降了，耗时升了

| | AIC 半边行数 | `wait_flag` | `set_flag` | `pipe_barrier` | 同步合计 | 中位数耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| base | 757 | 34 | 33 | **11** | 78 | **584.88** |
| `qk_manual` | 773 | 35 | 34 | **4** | **73** | **597.75（+12.88）** |

`pipe_barrier` 降 **64%**（11→4），同步合计降 6.4%，**而耗时变差 12.88 μs**
（6 样本同卡 ABBA）。

### 92.3 结论：撤回第 91 节，同步计数不是有效指标

1. **"同步数主导 387 μs scalar"不成立。** 若成立，减少 64% 的 barrier
   至少不该变慢。那个"约 46 周期/同步"的反推系数是把 387 μs 硬除出来的，
   没有独立依据——典型的用结果反推参数再拿参数解释结果。
2. **"同步计数是确定性指标"也不成立。** 它与耗时不相关，不能替代测量。
   第 91.6/91.7 提出它的初衷（绕开 10 μs 分辨力）是对的，但这个代理量无效。
3. 变差的可能原因（未验证）：`pl.unroll` 展开使代码略增（757→773 行）；
   手工 split-K 失去了 `AutoTileMatmulL0` 的流水编排（它能在 K 块之间做
   double buffer，手工版每块都要等 extract 完成）。**编译器的自动分块在这里
   优于手工**——与第 89.3 节"索要满核有害"、第 32 节同属一类：
   不要假定手工接管比框架的既有编排好。

### 92.4 本轮至此的总账

| 轴 | 候选 | 结果 |
| --- | --- | --- |
| 布局 | `wkv` / `cmp_wkv` / `cmp_wgate` → NZ | +14.75 / +20.25 / 符号不稳 ✗ |
| 核间流水 | `gather_full`（提并行度） | +19.00 ✗ |
| 核间流水 | `gather_pipe`（软流水） | +10.25 ✗ |
| 核内 | `qk_manual`（手工 split-K） | **+12.88 ✗** |

**六个候选全否，七档加权仍 1.117。** 并且第 58、90、91 三节提出的三个根因
（GM 带宽、L1 布局、同步数）**全部被自己后来的取证或实验推翻**。

现在可以确定的只有现象层面的事实：AIC scalar 占 59.2%（单核 387 μs）、
均摊无热点、mac/scalar 是 Native 的 0.55 倍。**这 387 μs 的构成仍未查明。**
下一轮不要再从"某个构造导致 scalar"出发猜，要么拿到真正的 PMU stall 分解
（区分标量指令执行 vs 等待），要么接受当前分解并把目标重新校准。

## 93. 工具链到顶：profiler 无法拆开 AIC 的 387 μs scalar（2026-09-29）

第 92.4 节要求"拿到真正的 PMU stall 分解（区分标量指令执行 vs 等待）"。
把 `torch_npu.profiler.AiCMetrics` 的模式全试过了，结论是**做不到**。

### 93.1 `ResourceConflictRatio` 只覆盖 AIV

`AiCMetrics` 的全部取值：`AiCoreNone`、`ArithmeticUtilization`、`L2Cache`、
`Memory`、`MemoryAccess`、`MemoryL0`、`MemoryUB`、`PipeUtilization`、
`ResourceConflictRatio`。其中只有最后一个与"停顿"相关，但它给出的列是：

```
aiv_vec_bankgroup_cflt_ratio
aiv_vec_bank_cflt_ratio
aiv_vec_resc_cflt_ratio
```

**三列全是 `aiv_` 前缀，没有任何 AIC 列。** 实测值（128K/B16 同卡两侧）：

| 侧 | bankgroup | bank | resc |
| --- | --- | --- | --- |
| PTO `aicore_kernel_mode_0` | 3.70% | 1.80% | 0.00% |
| Native `SparseAttnSharedkv` | 2.90% | 1.10% | 0.00% |

AIV 的冲突率本身就很低（< 4%）且两侧接近，**不是瓶颈**；而要查的 AIC 侧无数据。

### 93.2 因此 387 μs 在当前工具下不可再分

`PipeUtilization` 的 `aic_scalar_ratio` 是"标量单元非空闲的时间占比"，
它把**指令执行**与**等待**合在一起。要拆开需要：

- msprof 的原始 PMU 事件计数（超出本项目脚本的常规用法），或
- incore 模拟器的周期级 trace（`pypto` 的 `incore-profiling` 技能，
  但 HCA 算子需要 vllm 运行时，本机不具备）。

两者都不是当前工具链的常规路径。**在拿到其中之一之前，
任何"AIC scalar 由某构造导致"的说法都只能是假设**——
本轮已经有三个这样的假设（GM 带宽、L1 布局、同步数）先后被自己的取证或
实验推翻，不应再产生第四个。

### 93.3 本轮最终状态

| | |
| --- | --- |
| 七档加权 | **1.117**（目标 0.80，差 0.317），**未改进** |
| 候选 | **6 个全否**：`wkv`/`cmp_wkv`/`cmp_wgate` → NZ、`gather_full`、`gather_pipe`、`qk_manual` |
| 被推翻的根因假设 | **3 个**：GM 带宽（第 58 节←第 90 节）、L1 布局（第 90.10 节←第 90.11 节）、同步数（第 91 节←第 92 节） |
| 口径更正 | **7 次**（详见第 91.9 节末） |

确定的现象层事实（未被推翻）：

- AIC scalar 占 59.2%（平均单核 387 μs / 654 μs 墙钟），**均摊在 134 行循环体 ×
  144 次执行上、无单一热点**；
- 两侧 scalar 占比几乎相同（PTO 59.2% / Native 59.4%），而 Native 的 mac 是
  1.8 倍（26.7% / 14.9%）——**同样的标量预算推进了更多 MAC**；
- 已排除：GM 带宽（PTO mte2 仅 29.9%，Native 91.4%）、L1 容量与利用率
  （PTO 90.6% > Native 87.5%）、并行度、软流水、取模（0.5%）、
  AIV 资源冲突（< 4%）。

### 93.4 给下一轮的判断

在"算子侧 + 当前工具链 + 不改 PyPTO"这三个约束下，**0.80 没有可执行路径**：
六个候选、三个根因、以及本节的工具上限共同说明，剩下的差距落在
PyPTO 为这个循环生成的标量序列上，而那既看不见（工具不给分解）
也动不了（codegen 属"不做大修改"的范围）。

需要外部决策的两件事：

1. **放开改 PyPTO codegen**（同步插入／标量序列生成），并配套拿到周期级验证手段；
2. **重新校准 0.80** ——若差距主体是 DSL 生成对手写 AscendC 的固有差，
   该目标可能不在算子调优的可达范围内。第 63.3 节记的 `ops-transformer` 配方
   是手写实现，把它的绝对耗时当 PTO 的目标，前提是两者的代码生成质量可比。

## 94. ★★★★ 漏掉的一整层：`pypto.torch.init` 的运行时参数（2026-09-29）

第 93.4 节说"算子侧 + 当前工具链 + 不改 PyPTO 三个约束下 0.80 没有可执行路径"，
并且此前多次声称"编译开关已穷举"。**那是错的**——我只看了
`PassContext`（`memory_planner`、`enable_pypto_l0c_double_buffer`），
从没看 `pypto.torch.init` 这一层。它的签名里有一串未试过的参数：

```python
def init(device=None, platform="a2a3", runtime="tensormap_and_ringbuffer",
         aicpu_thread_num=0, enable_chip_swimlane=0, enable_dep_gen=False,
         output_dir=None, ring_task_window=None, ring_heap=None, ring_dep_pool=None)
```

**为什么这一层要紧**：第 87 节量到 `simpler_aicpu_kernel_exec`（683.1 μs）
比计算核（663.2 μs）还长、晚 18 μs 收尾，**device span 由它而非计算决定**，
31 个用例无一例外（AICPU 占 span 98.5%+）。前七个候选优化的都是那 663 μs 的计算，
而卡住 span 的是这 683 μs 的 AICPU 建图——`runtime` 与 `aicpu_thread_num`
正是作用在它上面的。

### 94.1 `aicpu_thread_num`：只有 4 可用，初测 −11 μs

文档写 "0 selects the runtime default; otherwise 2..5"，实测（128K/B16 同卡 ABBA）：

| 值 | 结果 |
| --- | --- |
| 2 | ✗ `RuntimeError: npuSynchronizeDevice ... NPUStream` 运行时崩溃 |
| **4** | **−11.00 μs**（3 样本，中位数 578.50 vs base 589.50） |
| 5 | ✗ `ChipWorkerError('simpler_kernel_mode_init failed with code -1000')` |

**合法区间名义是 2..5，实际只有 4 能跑。** −11.00 只有 3 样本、极差 39.25 μs，
正是 `cmp_wgate` 那种形态（第 88.12 节：四轮 −8/−13/−10.5/+2.75 最终判为不可用），
已补到 6 样本复测。

### 94.2 `runtime="host_build_graph"`：阻塞点具体且可修

必须在 `init` 传，不能经 `PassContext`——后者报
`Active PassContext runtime 'host_build_graph' conflicts with runtime
'tensormap_and_ringbuffer' bound by pypto.torch.init`。改到 `init` 后换成新错误：

```
hc_pre_fused.py:113:5: HBG kernel Host orchestration 'hc_pre_norm' cannot use
tensor.read on Tensor storage. Pass the required Host value as an explicit scalar argument.
```

即 `hc_pre_norm` 里这三行：

```python
scale0 = pl.read(hc_scale, [0])
scale1 = pl.read(hc_scale, [1])
scale2 = pl.read(hc_scale, [2])
```

HCA 包共 68 处 `pl.read`（compressor 15、hc_pre 6、decode_hca 3、sparse_attn 44），
但报错只点 **Host orchestration 层**那处，说明 spmd／incore 内的 read
（如 `pl.read(kv_seq_lens, [request])`）不受此限。

修法：把 `hc_attn_scale` 的三个值在主机侧读出，作为 `pl.Scalar[pl.FP32]`
参数经根签名传入。**要动根函数签名与 `native_adapter`（接口面），
工作量远大于 `aicpu_thread_num`**，留作 `at4` 定论之后的下一步。

潜在收益：若建图完全挪到主机，span 从 692 降到计算核的 663 附近，约 **−29 μs**
（约 5%，1.117 → 约 1.06）。不足以单独到 0.80，但会是本轮最大的确定性收益。

### 94.3 这是本轮第八次同类错误

前七次是口径搞错（把两 lane 当一个、循环边界算错、小样本当效应、
漏数 Mat 驻留物、误读 Native 流水级数、参数从结果反推），
这次是**搜索范围没覆盖就宣布穷举**。
教训：**说"穷举了"之前，先把配置面的每一层都列出来**——
`PassContext` 之外还有 `pypto.torch.init`，`init` 之外还有 `RunOptions`／
`CompileOptions`（第 08-entry-points.md 文档有完整清单）。

### 94.4 ✗ `aicpu_thread_num=4` 补测后符号翻转

3 样本时 −11.00 μs，补到 4 样本后 **+5.00 μs**
（base 中位数 576.75，`at4` 581.75）。与 `cmp_wgate`（第 88.12 节）同一形态，
效应在噪声带内，**不可用**。

### 94.5 ✗ `runtime="host_build_graph"`：三个阻塞全解开，但慢 157 倍

阻塞链（每一步都零接口改动）：

| 阻塞 | 位置 | 修法 |
| --- | --- | --- |
| `PassContext` 改不了 runtime | — | 改到 `pypto.torch.init` 传 |
| Host orchestration 不能 `tensor.read` | `hc_pre_fused.py:113`（`hc_pre_norm`） | 三个值只在各自 spmd 体内用 → **read 下移进 spmd** |
| 同上 | `hc_pre_fused.py:60`（`_hc_pre_partials`） | 查出是**死代码**（定义后从未使用）→ 删除 |

编译终于通过，但实测：

| 侧 | span |
| --- | --- |
| Native | 548.75 / 534.50 |
| **PTO（HBG）** | **90 923.5 / 94 586.0** |

**慢约 157 倍**，不是 −29 μs。

原因：HBG 文档写 "required for **Graph Execution**"，那指的是 **PyPTO 自己的
graph 模式**（`pl.graph` / `@pl.jit.graph`，见 pass 08「outline_graph_scopes」），
**不是 torch_npu 的 NPUGraph**。本项目的计时口径是把 PTO 调用捕获进 NPUGraph
再重放（`measure_graph_interval`），在这条路径上 HBG 拿不到它需要的 host 端图，
于是退化成每次调用重建整张任务图——90 ms 正是那个量级。

**`host_build_graph` 与当前 NPUGraph 回放口径不兼容，否决。**

顺带的收获：HBG 的 Host-orchestration 检查暴露了 `_hc_pre_partials` 里三行
死代码（读了 `hc_scale` 却从不使用）。这处**可以独立清理**，与 HBG 无关。

### 94.6 本轮最终账：9 个候选全否

| 轴 | 候选 | 结果 |
| --- | --- | --- |
| 布局 | `wkv` / `cmp_wkv` / `cmp_wgate` → NZ | +14.75 / +20.25 / 符号不稳 ✗ |
| 核间流水 | `gather_full` | +19.00 ✗ |
| 核间流水 | `gather_pipe` | +10.25 ✗ |
| 核内 | `qk_manual`（手工 split-K） | +12.88 ✗ |
| 运行时 | `aicpu_thread_num` 2 / 5 | 崩溃 / 初始化失败 ✗ |
| 运行时 | `aicpu_thread_num` 4 | −11.00 → **+5.00** 符号翻转 ✗ |
| 运行时 | `runtime=host_build_graph` | 编译通了但 **慢 157 倍** ✗ |

**七档加权仍 1.117，本轮零改进。**

三个被推翻的根因假设（GM 带宽、L1 布局、同步数）+ 九次判断错误
（七次口径、一次搜索范围、一次把 PyPTO 的 graph 当成 NPUGraph）。

现在可以说的是：**算子侧调参、布局、并行度、软流水、以及 `init`／`PassContext`
两层的运行时参数都已实测穷举**（这次是真的把配置面列完了：
`docs/zh/dev/08-entry-points.md` 的 `RunOptions`／`CompileOptions`／`DfxOptions`
三张表 + `init` 签名 + `PassContext` 签名）。剩下的差距落在 PyPTO
为这个 kernel 生成的代码质量上，而那既缺乏可测手段（第 93 节：profiler
分不出标量执行与等待），也属"不做大修改"的范围。

## 95. ★★★★★ 连删除死代码都慢 21.88 μs：这个算子处在被扰动即退化的局部最优（2026-09-29）

第 94.5 节顺带发现 `_hc_pre_partials` 里三行 `pl.read(hc_scale, [i])`
读了从不使用（脚本里加断言核对过函数体 63–94 行确无引用）。它是 Host
orchestration 层的 GM 标量读，按理删掉只会更快或无变化。

实测（128K/B16 同卡 ABBA，各 6 样本）：

| 变体 | 样本 | 中位数 | 相对 base |
| --- | --- | --- | --- |
| base | 583.8 / 574.8 / 580.5 / 577.5 / 578.5 / 550.8 | 578.00 | — |
| `deadread`（删那三行） | 577.8 / 598.2 / 597.8 / 617.0 / 617.5 / 601.5 | 599.88 | **+21.88 ✗** |

样本分离清晰（`deadread` 落在 577~617，base 落在 550~583），不是噪声。

### 95.1 这说明什么

那三行不是"无副作用的死代码"。合理的解释有两类，都未单独验证：

1. **它们有实际作用**——例如把 `hc_scale` 带进 cache，或在 Host 层造成一次
   恰好有利的同步/排序；
2. **删除改变了 IR，从而改变了下游 pass 的决策**——任务划分、同步插入、
   内存规划都可能因此走到另一个分支。第 92 节已经见过同类现象：
   `qk_manual` 把 `pipe_barrier` 降了 64%，耗时反而 +12.88 μs。

无论是哪一类，结论一致：**这个算子对 IR 的任何扰动都敏感，而扰动的方向
几乎总是变慢。**

### 95.2 本轮十个候选的分布，本身就是结论

| 类别 | 候选 | 结果 |
| --- | --- | --- |
| 布局 | `wkv` / `cmp_wkv` / `cmp_wgate` → NZ | +14.75 / +20.25 / 符号不稳 |
| 核间流水 | `gather_full` / `gather_pipe` | +19.00 / +10.25 |
| 核内 | `qk_manual` | +12.88 |
| 运行时 | `aicpu_thread_num` 2 / 4 / 5 | 崩溃 / 符号翻转 / 初始化失败 |
| 运行时 | `runtime=host_build_graph` | 慢 157 倍 |
| 代码清理 | `deadread` | **+21.88** |

**十个候选，没有一个为负；范围从 +10 到 +157 倍。** 加上第 58 节的六个、
第 68 节的两个，累计 18 个候选里只有一次落地过 −47 μs
（`allow_early_resolve` ×16 + `QPROJ_WORKERS` 24→20，第 72 节）。

这不是候选挑得差——**是这一版算子已经被调到了一个尖锐的局部最优**，
第 58.1 节当时的判断（"算子已处在局部最优"）现在有十个新数据点支持它。
而且连"删掉不用的代码"这种通常无害的动作都要付 21.88 μs，
说明可动空间不只是小，而是**任何方向都在下坡**。

### 95.3 因此本轮到此为止，以及为什么

七档加权 **1.117**，目标 0.80，差 0.317，本轮零改进。

已实测穷举的面（这次把清单核全了）：
算子侧的分块／tiling／并行度／预取／软流水／权重布局／手工 L0 管理，
`PassContext` 与 `pypto.torch.init` 两层的全部运行时参数
（依据 `docs/zh/dev/08-entry-points.md` 的三张表加两个函数签名）。

未探但需要外部决策的两条：

1. **改 PyPTO 的代码生成**（同步插入粒度、标量序列）。第 90.2 节的
   mac/scalar 比（Native 0.45 / PTO 0.25）说明同样标量预算下 Native 推进 1.8 倍
   MAC，差距在生成质量上。但第 93 节已确认 **profiler 分不出标量的执行与等待**，
   所以这条路必须同时解决"怎么验证"——否则又会重复本轮"改一版、测一版、
   符号随机"的循环。
2. **重新校准 0.80**。`ops-transformer` 是手写 AscendC。把它的绝对耗时当 PTO
   的目标，前提是两者代码生成质量可比；本轮的十个否决与 mac/scalar 的差距
   都在质疑这个前提。

## 96. ⛔ HBG 路径按用户裁定停止；`readdown` 也否（2026-09-29）

**用户 2026-09-29 裁定：「HBG 的路径停掉，之后我会安排做」。**
第 94.5 节那条 `runtime=host_build_graph` 的线索到此为止，不再由本项目推进
（三个阻塞的解法与那 157 倍退化的原因都已记在 94.5，交接时可直接用）。

### 96.1 ✗ `readdown`：把 Host 层标量读下移进 spmd

这是 HBG 过程中做出来、但**与 HBG 无关**的独立候选：`hc_pre_norm` 里三行
`pl.read(hc_scale, [i])` 原在 Host orchestration 层（由 AICPU 执行），
而泳道显示 device span 正由 AICPU 决定（第 87 节），所以把它们下移到各自
spmd 体内、改由计算核承担。

6 样本同卡 ABBA：base 中位数 578.50，`readdown` 591.00 → **+12.50 μs**。

与 `deadread`（+21.88，第 95 节）方向一致：**对这个算子的任何 IR 扰动都变慢**，
即便扰动本身在直觉上是"把工作挪离瓶颈"或"删掉不用的代码"。

### 96.2 本轮十一个候选的完整清单

| 类别 | 候选 | Δ vs base |
| --- | --- | --- |
| 权重布局 | `wkv` / `cmp_wkv` / `cmp_wgate` → NZ | +14.75 / +20.25 / 符号不稳 |
| 核间流水 | `gather_full` / `gather_pipe` | +19.00 / +10.25 |
| 核内 | `qk_manual` | +12.88 |
| 运行时 | `aicpu_thread_num` 2 / 4 / 5 | 崩溃 / 符号翻转 / 初始化失败 |
| 运行时 | `runtime=host_build_graph` | 慢 157 倍（**已按用户裁定停止**） |
| 代码清理 | `deadread`（删三行未使用的 read） | +21.88 |
| Host→spmd | `readdown`（标量读下移） | **+12.50** |

**十一个候选无一为负。** 七档加权仍 **1.117**。

## 97. ★★★★★ 换成 min/mean/max 口径后，本轮十个"已否决"全部变成"不可判定"（2026-09-29）

用户要求「后续的时间看最大，最小，平均值三个」。改了
`summarize_hca_ab.py` 之后，第一次把候选的 Δ 与 **base 自身的极差**放在一起看，
结论整体翻转。

### 97.1 复核结果

| 候选 | Δ(平均) | base 极差 | 判定 |
| --- | ---: | ---: | --- |
| `nz_wkv3` | +8.29 | 22.75 | 不可判定 |
| `nz_cmpwkv` | +15.62 | 18.50 | 不可判定 |
| `nz_cmpwgate` | −9.54 / +2.17 | 21.75 / 30.00 | 不可判定 |
| `gather_full` | +7.38 | 31.50 | 不可判定 |
| `gather_pipe` | +7.12 | 40.00 | 不可判定 |
| `qk_manual` | +11.71 | 29.25 | 不可判定 |
| `at4` | −4.94 / +9.96 | 33.75 / 35.50 | 不可判定 |
| `deadread` | +27.33 | 33.00 | 不可判定 |
| `readdown` | +7.88 | 15.25 | 不可判定 |
| `nowarm_cmp` | −7.58 / +21.88 | 33.75 / 42.75 | 不可判定 |
| **`syncstart`（121 处全局开 `sync_start=True`）** | **+66.00** | 37.00 | **明确变差** |

**base 自己的极差是 15~43 μs，而候选的 Δ 只有 ±0.4~27 μs。**
此前用「6 样本中位数」报的 +14.75／+20.25／+16.00 等全部埋在噪声里，
那十个"已否决"的结论**都不成立**，正确说法是不可判定。

这也解释了此前无法理解的现象：同一候选多轮符号翻转
（`nz_cmpwgate` 四轮 −8.0/−13/−10.5/+2.75，`at4` 两轮 −11.0/+5.0）。
不是测量出错，是效应本来就小于噪声。

**第 68.1 节的「≥6 样本才能定符号」不够**——样本数只让均值稳定，
不会让 10 μs 的效应从 40 μs 的抖动里浮出来。判据要加一条：
**`|Δ|` 必须明显超过 base 的极差。** 脚本现在会自动标注。

### 97.2 ✗ `sync_start=True` 全局开启：+66 μs，这是本轮唯一超出噪声的结果

用户给的两条 SPMD 原则之一。盘点发现**全仓 121 处 `pl.spmd` 无一设置
`sync_start`**，于是全部加上 `sync_start=True`（纯机械改动，零语义变化）。

结果 **+66.00 μs**（base 极差 37.00），远超底噪，**是可信的负面结论**。

机制：`sync_start=True` 要求所有块同时开始，等于把 gang 调度收紧——
必须等够数的核同时空出来才能启动。而这个算子的任务图恰恰靠"块陆续上核"
填补空隙，第 72 节落地的 −47 μs 正是 `allow_early_resolve` ×16 带来的
（让任务在依赖尚未全部就绪时提前解析上核）。`sync_start` 把这个弹性拿掉了。

与第 32 节「索要满核有害」、第 89.3 节 `gather_full`(+19) 同属一个机制的三个实例：
**这个算子的性能来自调度弹性，任何收紧同步或强求整齐的改动都在下坡。**

⚠ 注意这不否定用户原则本身在一般情况下的正确性——`available_*_count()` +
`sync_start` 是 PyPTO 文档推荐的组合，而且文档写明
「A hard `pl.system.syncall` needs full occupancy to complete, so this is the
only launch width that keeps it deadlock-free across devices」，
即用满核还带**跨设备防死锁**的正确性含义。本节只证明：
在 HCA 当前这版任务图上，全局开 `sync_start` 的性能代价是 +66 μs。

### 97.3 原则 1（用满核）的盘点结果

块数全是语义推导值，**没有一个用 `available_cluster_count()` /
`available_aiv_count()`**。泳道实测（128K/B16）AIV 侧浪费：

| task | 实测块数 | 满核应为 | 浪费 |
| --- | ---: | ---: | ---: |
| `kv_rms_norm_rope` | 3 | 48 | 45 |
| `hca_warm_kv_weights`、`hca_raw_cache_write` | 8 | 48 | 40 |
| `mix_x_rms_norm`、`comb_sinkhorn`、`qr_rms_norm_quant`、`hca_hc_widen_rms` | 12 | 48 | 36 |
| `hca_softmax_pool`、`hca_state_commit`、`hca_gather_kv`、`hca_inverse_rope_sign`、`hca_norm_rope_write` | 16 | 48 | 32 |
| `quant_0` | 24 | 48 | 24 |
| `qproj_matmul` | 20 | 24（AIC） | 4 |
| `kv_proj_matmul` | 12 | 24（AIC） | 12 |
| `hca_kv_score_proj` | 16 | 24（AIC） | 8 |

这与第 87 节量到的「AIV 只用到上限 79%」一致。但要把这些提到满核，
需要重写每块的工作划分（例如 `hca_softmax_pool` 现在每块一个请求，
B16 只有 16 块；用满 48 核得改成请求×列块的二维分工），
属于逐个改、逐个测的工作，且每项都要按 97.1 的新判据定符号。

## 98. SPMD 满核 + `sync_start`：`sync_start` 必须与满核配套，且提块数不能压缩核内流水（2026-09-29）

用户 2026-09-29 给出两条 SPMD 原则（尽可能用满核 + 设 `sync_start`），
随后把目标收窄为「**在尽可能用满核又不影响核内流水的前提下**叠加 `sync_start`，
逐个逐个改，有并发的需要联动改」。本节记录前三个数据点与由此得到的判据。

### 98.1 三个数据点分离出 `sync_start` 的真实作用

| 变体 | 满核 | `sync_start` | 影响核内流水 | Δ(平均) | base 极差 | 该变体极差 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `syncstart`（121 处全开） | ✗ | ✓ | — | **+65.96 有害** | 35.75 | 35.75 |
| `quant48`（24→48 块） | ✓ | ✗ | ✓ 流水 4→2 次 | +1.46 不可判定 | 46.25 | **27.25** |
| `quant48ss` | ✓ | ✓ | ✓ 同上 | +2.29 不可判定 | 44.25 | **23.00** |

两条结论：

1. **`sync_start` 不能脱离满核单独使用。** 同样开 `sync_start`：在未满核的
   121 处是 **+66 μs**，在满核的 `quant` 上只有 +0.83 μs（`quant48ss` 相对
   `quant48`）、落在噪声内。机制是 `sync_start` 要求所有块同时开始，
   未满核时这纯粹增加启动等待，拿不到满核的补偿。
   **用户的原则成立，我此前只做后半条的做法是错的。**
2. **极差随满核单调收窄**：44~46 → 27.25 → 23.00。满核 + 同步启动让该任务
   不再受其他任务占核状态影响，波动几乎减半。这本身有价值（可预测性），
   但没转化成平均耗时。

### 98.2 ★ 判据：提块数不能靠压缩内层循环

`quant` 没拿到收益的原因是 `QUANT_TASK_T_TILE` 32→16 把内层
`pl.pipeline` 从 4 次迭代压到 2 次——**用流水深度换了核数，两者抵消**。
这正是新目标里「不影响核内流水」要排除的。

于是 SPMD 任务分成两类：

| 类型 | 结构 | 提块数的后果 |
| --- | --- | --- |
| **单维分块** | 块数 × 内层次数 = 常量（`raw_cache_write`：`pl.range(blk, tokens, CACHE_WORKERS)`；`raw_valid`：`pl.range(min(VALID_TOKEN_TILE, …))`；`inverse_rope_sign`：按 `ROPE_CS_T_TILE` 分段；`quant`：`QUANT_TASK_T_TILE`） | **必然压缩内层，不可做** |
| **有第二个独立维度** | 并行维与核内维正交（`softmax_pool`：`h0` 4 段 × `history_row` 128 次） | **把第二维提到并行维度，内层一行不动** |

**只有第二类值得改。** 第一类要改，必须先找出它自己的第二个可并行维度
（例如 `raw_cache_write` 的 HEAD_DIM 方向），否则就是重复 `quant48` 的错误。

### 98.3 `pool48`：第一个三条件全满足的候选

`hca_softmax_pool` 原本每块一个请求，B16 只有 16 块、48 个 AIV 核空 32 个；
外层 `for request` 只迭代一次，**本就不承载流水**。改法：

```python
pool_segments = HEAD_DIM // POOL_COLS            # 4
pool_units    = requests * pool_segments         # B16: 64
pool_workers  = pl.min(pool_units, pl.system.available_aiv_count())   # 48，满核
with pl.spmd(pool_workers, …, sync_start=True):
    for unit in pl.range(pl.tile.get_block_idx(), pool_units, pool_workers):
        request = unit // pool_segments
        h0      = pl.max(unit - request * pool_segments, 0) * POOL_COLS
        …
        for history_row in pl.range(RATIO):      # 128 次，一行未动
```

用 `available_aiv_count()` 而非字面量 48：核数属于运行落到的设备，
写死会 under/over-fill，且 `pl.system.syncall` 需要 full occupancy
才不跨设备死锁（PyPTO 文档 `system_ops.py`）。

量级：`softmax_pool` 814.9 核·μs 现摊在 16 块（墙钟 52.5 μs），
48 块理论约 17 μs。已提交 8 样本，编译通过（首个 pool48 pass 589.25 vs base 613.5）。

### 98.4 簇划分与联动范围

按泳道（128K/B16）的时间重叠划簇：

| 簇 | 时段 | 成员（块数） | 并发峰值 | 空闲核 |
| --- | --- | --- | ---: | ---: |
| A | 0–100 | widen_rms(12)、warm_kv(8)、q_rope_prepare(12)、mix_x_rms(12)、comb_sinkhorn(12) | 32 | 16 |
| B | 120–213 | qr_rms_quant(12)、kv_rms_rope(3)、**softmax_pool(16)**、raw_cache_write(8)、inverse_rope(16)、raw_valid(12) | 32 | 16 |
| C | 211–245 | gather_kv(16)、state_commit(16)、norm_rope_write(16) | **48** | **0（已满，不动）** |
| D | 499–576 | quant(24) | 24 | 24 |

**簇内必须联动**（用户 2026-09-29 的判据）：两个 12 块任务并发时共占 24 核，
只把一个提到 48 会让它等不到核；两个都提则各自跑满、反而快。

已确认不可动的硬约束：`T_TILE = 8`（源码注明 `other values miscompare`）、
`LINEAR_T_TILE = 16`（cube 行必须 16 行整块）、`WIDEN_ROWS = 8`
（`pl.row_sum` 的 FP32 结果至少需 8 个物理行，见第 40.1 节）。

### 98.5 ★ 判据补充：max 下降本身就是收益（多卡 MoE）

用户 2026-09-29 指出：**「最大值减小也是一个有益的收益，因为多卡并行时，
最大值会影响其他卡的 MoE 通信性能。」** 多卡部署里每张卡跑完 attention 半边后
进入 MoE 的 all-to-all，集合通信是同步的——**所有卡到齐才能开始**，
于是单卡的尾延迟直接传导成其他卡的等待。

按这条重新复核全部 20 个数据点，结论换了次序：

| 候选 | Δ平均 | **Δmax** | **Δ极差** |
| --- | ---: | ---: | ---: |
| **`pool48`**（满核，不压流水） | +13.61 | **−4.75** | **−30.00** |
| **`quant48`**（满核，压了流水） | +4.94 | **−4.25** | **−19.00** |
| `syncstart`（未满核却开 sync_start） | +65.96 | **+57.50** | −9.25 |
| `deadread` | +27.33 | **+33.75** | +6.75 |
| `qk_manual` | +11.71 | **+22.75** | +13.25 |

**20 个数据点里唯二同时降 max 且极差收窄 >19 μs 的，正是唯二的满核改造。**
其余 18 个的极差收窄都不到 19。核占满后该任务不再受其他任务占核状态影响，
尾部被削平——这不是平均值能反映的。

⚠ 相应地，此前只看平均值的判定要补一列 max。抬高 max 的改动
（`qk_manual` +22.75、`deadread` +33.75、`syncstart` +57.50）在多卡上
比平均值显示的更糟。

### 98.6 ✗ `sync_start` 即使在满核任务上也不划算

三方同卡对照（`base` / `pool48` = 满核+sync_start / `pool48ns` = 满核不开）：

| | 平均 | max | 极差 |
| --- | ---: | ---: | ---: |
| base | 594.81 | 610.75 | 30.50 |
| `pool48`（满核 + `sync_start`） | 601.25 | 616.00 | 25.25 |
| **`pool48ns`（满核，不开）** | 599.19 | **607.50** | **15.50** |

`pool48ns` 三项全优于 `pool48`（平均 −2.06、max −8.50、极差 −9.75），
且是三者里唯一把 max 压到 base 以下的，极差只有 base 的一半。

与 `quant` 上的结果对齐后，两个满核任务的结论一致：

| 任务 | 满核不开 | 满核开 | `sync_start` 净作用 |
| --- | --- | --- | ---: |
| `quant` | Δmax −4.25，极差 27.25 | Δmax +3.50，极差 23.00 | **max +7.75** |
| `softmax_pool` | max 607.50，极差 15.50 | max 616.00，极差 25.25 | **max +8.50** |

**`sync_start` 在两个满核任务上都把 max 抬高约 8 μs。** 第 98.1 节曾看到
"满核后 sync_start 只 +0.83 μs、落在噪声内"，那是只看平均值的错觉；
按 max 看它始终是负的。此前 `pool48` 一轮 6 样本 Δmax −4.75、这轮 4 样本 +5.25
的矛盾也来自样本不足。

**修正结论**：真正带来收益的是「用满核」这一条。`sync_start` 在未满核时是
+66 μs 的灾难，满核后代价降到约 +8 μs 的尾延迟，**但仍未找到它能加分的场景**。
后续候选默认不开 `sync_start`，除非某个任务上能实测出它降 max。

### 98.7 ✗ AIC 侧满核：`kvscore24` 反向，`qproj_matmul` 早已被否

按"AIC 是瓶颈资源"（下限 474.8 vs AIV 380.0）转向 AIC 侧填满核，两个候选都不成立。

**`kvscore24`**：`hca_kv_score_proj` 块数 = `ceil(tokens/MM_ROWS) × (HEAD_DIM/MM_COLS)`
= 2×8 = 16，AIC 满核是 24。把 `MM_ROWS` 64→32 得到 3×8 = 24 正好满核，
内层 K 方向 `pl.pipeline` 不动。实测（4 样本同卡）：

| | 平均 | max | 极差 |
| --- | ---: | ---: | ---: |
| base | 578.38 | 593.25 | 28.00 |
| `kvscore24` | 589.06 | **617.25** | **52.50** |

**max 抬高 24 μs、极差从 28 涨到 52.5**，与前两个满核改造（降 max、收窄极差）
完全反向。原因：为凑满 24 块把 `MM_ROWS` 从 64 砍到 32，**matmul 的 M 维减半
直接伤 cube 效率**。这与 `quant48` 同构——都是盯着块数填满、却拿另一个维度的
质量去换（`quant` 换流水深度 4→2，这次换 cube 的 M 维 64→32）。

**`qproj_matmul`**：想把 `QPROJ_WORKERS` 改回 24 满核，但生产值是 **20**，
且注释写明「留余量：`QPROJ_N_BLOCKS=128` 时 24 个 worker 只比核数少一点，
派发时若有核被占就要等」。第 72 节记录它是那次 **−47 μs 落地的组成部分**
（`allow_early_resolve` ×16 + `QPROJ_WORKERS` 24→20），第 61 节还实测过
24→32/48 七档加权 +9.33/−2.28 均无效。

**这是「用满核」的一个实测反例：24 才是满核，但 20 更快**——留余量让任务
不必等最后几个核空出来。

### 98.8 ★ 修正后的适用条件：只有"分工余数"型的不满核值得填

三次失败（`quant48` 压流水、`kvscore24` 缩 M 维、`qproj` 满核反而慢）
与两次成功（`pool48` / `pool48ns` 降 max 收窄极差）对比，条件收窄为：

| 不满核的原因 | 能否填满 | 例子 |
| --- | --- | --- |
| **有独立的第二并行维未被利用** | ✓ **值得填** | `softmax_pool`：`h0`(4 段) 与 `history_row`(128 次) 正交，提 `h0` 后内层一行不动 |
| 块数 × 内层次数 = 常量（单维分块） | ✗ 必压内层流水 | `quant`、`raw_cache_write`、`raw_valid`、`inverse_rope_sign` |
| 块数由 matmul 的 M/N 分块决定 | ✗ 必缩 tile、伤 cube 效率 | `kv_score_proj`、`kv_proj_matmul` |
| 已为"留派发余量"故意调低 | ✗ 填满反而更慢 | `qproj_matmul`（20 优于 24，第 72 节） |

**AIC 侧的任务全是 matmul，块数由 M/N 分块决定，所以 AIC 基本没有可填的空间**；
可填的只在 AIV 侧那些"每块一个请求、另有独立维度"的任务上。

### 98.9 ★ 泳道全量填充率：与资源下限的 192.9 μs 差距，全部是依赖阻塞

口径：`results/hca_swimlane_v2_20260929/.../hca_pto_nz2_early_qw20_128K_B16__merged_swimlane.json`，
**只取 pid 4 的 Worker View**（此前一次统计误把 Scheduler View 的 pid 2/3 也算进去，
块数虚高一倍——`warm_kv` 看成 16 块、`state_commit` 看成 6 块，均为错读）。

| | 核·μs | 下限 |
| --- | ---: | ---: |
| AIC（24 核） | 11 403 | **475.1** |
| AIV（48 核） | 18 242 | 380.0 |
| span | | **668.1** |

打包效率 **71.1%**，理论可省 **192.9 μs**。按 0.5 μs 网格把时间轴分成四类：

| 类别 | 时长 | 占比 |
| --- | ---: | ---: |
| 两侧都满 | 168.5 | 25.2% |
| 只有 AIC 满（AIV 空） | 222.5 | 33.3% |
| 只有 AIV 满（AIC 空） | 86.0 | 12.9% |
| **两侧都不满** | **191.5** | **28.6%** |

**「两侧都不满」的 191.5 μs 与理论可省的 192.9 μs 几乎相等。** 也就是说
与资源下限的全部差距都落在依赖阻塞窗口里，不在"某个任务块数不够"上。六个窗口：

| 窗口 | 时长 | AIC 均 | AIV 均 | 内容 |
| --- | ---: | ---: | ---: | --- |
| 0–39.5 | 39.5 | 1.4 | 3.8 | 启动（AICPU 建图，已知） |
| 81–98 | 17.0 | 3.6 | **32.0** | 簇 A：mix_x(12)+comb(12)+warm(8) |
| 186–199 | 13.0 | 16.8 | 19.0 | qproj/kv_score/softmax_pool 交接 |
| **206.5–285.5** | **79.0** | 13.6 | 22.7 | 簇 B+C 的依赖链，最大一段 |
| 595–632 | 37.0 | 10.0 | 4.5 | proj_b 收尾流入 oproj_hc_post |
| 662.5–668.5 | 6.0 | 0.0 | 20.4 | 尾巴 |

注意 340–500 的 attention 段（160 μs）**两侧恒为 24/48，已 100% 填满**，
没有任何块数可加。

### 98.10 ★ 填核的反作用力：每加一块要付约 0.27 μs 的 AICPU 派发

把五个已测的填核变体按「Δ块数 → Δ平均耗时」排开：

| 变体 | Δ块 | Δ平均 | μs/块 |
| --- | ---: | ---: | ---: |
| `pool48`（softmax_pool 16→48） | +32 | +13.61 | 0.425 |
| `pool48ns`（同上不开 sync_start） | +32 | +4.38 | 0.137 |
| `quant48`（quant 24→48） | +24 | +4.94 | 0.206 |
| `gather_full`（gather 32→64） | +32 | +19.00 | 0.594 |
| `kvscore24`（kv_score 16→24） | +8 | +10.68 | 1.335 |
| **加权** | **+128** | **+52.61** | **0.411** |

除 `kvscore24`（它同时把 cube 的 M 维减半，不是纯填核）外，四个纯填核变体
落在 0.137~0.594 μs/块，与既有记录的 PyPTO 图重放开销 **每块 0.27 μs**
（另有每任务 2.3 μs）同量级。而泳道早已证明 span 由 AICPU 决定
（`simpler_aicpu_kernel_exec` 683.1 μs > 计算核 663.2 μs，且晚 18 μs 收尾）。

**所以「尽可能把核用满」在这条链上有一个固定标价：Δ耗时 ≈ 墙钟节省 − 0.27×Δ块数。**
填核净收益的必要条件是三条同时成立：

1. 该任务**在关键路径上**（否则省下的墙钟根本不进 span）；
2. 它所在窗口的瓶颈资源**就是它占的那种核**（AIC 满时加 AIV 块毫无用处）；
3. 省下的墙钟 **> 0.27 × Δ块数**。

复盘 `pool48`：它把 softmax_pool 的墙钟从 52.5 压到理论 17 μs，本该净赚 27 μs，
但实测 +4.38~+13.61。原因是第 2 条不成立——176–228 那段 **AIC 才是瓶颈**
（占用 20~24），AIV 有 24 个核闲着，压缩 AIV 上的任务不改变 span。
它唯一兑现的是 max 与极差（第 98.5 节），那是多卡收益、不是单卡收益。

### 98.11 ★ 起点跨度把任务分成两类，解释了 `sync_start` 为何总是负的

同一份泳道，按「块的起跑时刻跨度」排序：

| 任务 | 块 | 墙钟 | 块均 | **起点跨度** |
| --- | ---: | ---: | ---: | ---: |
| `hca_softmax_pool` | 16 | 52.5 | 50.93 | **0.2** |
| `hca_raw_cache_write` | 8 | 18.4 | 17.87 | **0.1** |
| `hca_inverse_rope_sign` | 16 | 20.2 | 19.06 | **0.2** |
| `kv_rms_norm_rope` | 3 | 51.1 | 50.79 | **0.1** |
| `hca_gather_kv` | 16 | 24.2 | 14.35 | 14.8 |
| `hca_state_commit` | 16 | 28.4 | **8.36** | **20.6** |
| `hca_kv_score_proj` | 16 | 45.9 | 27.53 | 24.4 |
| `hca_cmp_work_gather` | 48 | 80.3 | 23.14 | 61.9 |

**`softmax_pool` 的 16 个块在 0.2 μs 内全部起跑——它本来就是同步启动的，
`sync_start` 只能加一道屏障、不可能提速。** 这正是第 98.6 节测到
`sync_start` 在它身上 +8.5 μs max 的原因，而不是"sync_start 本身没用"。

反过来，`state_commit` 的墙钟 28.4 μs 里只有 8.36 是块内执行、20.6 是起点跨度，
且那一刻 AIV 只有 **7/48** 在跑——不是核被占满，是块一个个地进来。
**这才是 `sync_start` 唯一有机会加分的形态**，本轮据此设计了 `sc48ss`。

另外发现：`kv_rms_norm_rope` **已经开着 `sync_start=True`**，是生产代码里唯一一处。

### 98.12 ✓ `sc48`：第一个三项全负的候选（待确认轮）

`hca_state_commit` 从「每块一个请求」(16 块) 展平成「每块一个 token」
(requests × DECODE_SEQ = 96 单元 → 48 块)。`token = request*DECODE_SEQ + step`
与展平后的 unit 编号完全一致，所以 `token = unit`、`request = unit // DECODE_SEQ`，
store 的地址与数值逐 bit 不变（六轮 `精度=OK`）。

同卡 ABBA（device 2，CYCLES=3，每侧 6 样本）：

| | 样本 | 最小 | 平均 | 最大 | 极差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 6 | 562.25 | 582.25 | 608.25 | 46.00 |
| **`sc48`** | 6 | **558.50** | **573.42** | **590.00** | **31.50** |
| `sc48ss`（+`sync_start`） | 6 | 583.00 | 598.54 | 613.50 | 30.50 |

`sc48` 三项全负：平均 −8.83、**max −18.25**、极差 −14.50。按平均值判据
（Δ 8.83 < base 极差 46）仍是"不可判定"，但**逐名次差全部同号**：

```
sc48 − base 逐名次: −3.7  −5.0  −12.5  −6.4  −7.0  −18.2
```

Mann-Whitney U = 23.5（n=6 时显著需 ≥31），方向一致但样本不足，
已提交 `abba27_sc48conf`（CYCLES=5，每侧 10 样本）确认。

**这是本轮唯一一个满足第 98.10 节三条件的候选**：`state_commit` 在簇 B→C
的依赖链上（关键路径）、它所在窗口 AIV 只有 7/48（瓶颈就是 AIV）、
墙钟 28.4 → 理论 2.8 μs 的节省远大于 32 块 × 0.27 = 8.6 μs 的派发代价。

### 98.13 ✗ `sync_start` 在起点跨度最大的任务上仍然为负——问题就此定死

`sc48ss` 与 `sc48` 的唯一差别是 `sync_start=True`：

```
sc48ss − base 逐名次: +20.8  +19.3  +7.7  +25.0  +19.8  +5.3
```

U = 7.5，六个名次全部更慢，Δ平均 **+16.29**、Δmax +5.25。
相对 `sc48` 本身是 **+25.12 μs**。

至此 `sync_start` 在三种形态下全部为负：

| 形态 | 变体 | Δ平均 | Δmax |
| --- | --- | ---: | ---: |
| 未满核、121 处全开 | `syncstart` | +65.96 | +57.50 |
| 满核、起点跨度 0.2 μs | `pool48` vs `pool48ns` | +2.06 | +8.50 |
| 满核、**起点跨度 20.6 μs**（最有理由的一处） | `sc48ss` vs `sc48` | **+25.12** | **+23.50** |

第 98.11 节曾推测"起点跨度大的任务是 sync_start 唯一有机会加分的形态"，
**这条推测被自己设计的实验否掉了**：跨度最大的 `state_commit` 上它反而最贵。

推断机制：`sync_start` 要求所有块同时开始，于是整个任务要等到最后一个核
空出来；而起点跨度大恰恰说明核是陆续空出来的，等齐的代价正比于跨度。
跨度 0.2 μs 时代价 8.5 μs（纯屏障开销），跨度 20.6 μs 时代价 23.5 μs
（屏障开销 + 等齐）。**跨度越大越不该开，与直觉相反。**

结论：**后续候选一律不开 `sync_start`**，本项目不再为它设计实验。

### 98.14 ✗ `krr8` 与 `warm24`：块数/核数比最低的两个任务，填满都无效

同卡 ABBA（device 3，CYCLES=3，每侧 6 样本）：

| | 样本 | 最小 | 平均 | 最大 | 极差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 6 | 567.75 | 587.92 | 600.00 | 32.25 |
| `krr8`（`KV_RMS_T_TILE` 32→8，3 块→12 块） | 6 | 572.50 | 590.08 | **612.50** | 40.00 |
| `warm24`（`WARM_WORKERS` 8→24） | 6 | 557.00 | 586.83 | **612.25** | **55.25** |

两者的 Mann-Whitney **U 都恰好是 16/36**（36 对里正中间就是 18），
逐名次差符号混杂，**没有可检出的效应**；而两者都把 max 抬高约 12 μs，
`warm24` 的极差还从 32.25 涨到 55.25。

失败原因正是第 98.10 节的第 1、2 条：

- **`kv_rms_norm_rope`**：它跑在 155–206，那段 `qr_proj_matmul` 把 AIC 占满
  （24/24），**瓶颈是 AIC 不是 AIV**，压缩一个 AIV 任务的墙钟进不了 span。
- **`hca_warm_kv_weights`**：它**根本不在关键路径上**——预热只写哨兵，
  没有任何消费者，存在的意义就是占用本来空闲的 AIV 核。把它从 8 块提到
  24 块只是让同样的 527 核·μs 挤进更短的窗口，span 不变，反而多付
  16 块 × 0.27 = 4.3 μs 的派发、并把抖动放大（极差 +23）。

**教训**：「块数/核数比最低」不是挑候选的判据。`kv_rms_norm_rope` 3 块、
`warm_kv` 8 块，是全 kernel 最"空"的两个任务，填满却都无效——
判据只能是第 98.10 节那三条，其中**第一条（在关键路径上）最容易被忽略**，
因为泳道上"核空着"看起来总像是可以利用的。

### 98.15 `commit48cap`：一个零假设对照

`decode_compressor_ratio128.py` 的 `WORKERS = 16` 只经由
`pl.min(requests, WORKERS)` 使用，B16 下 `min(16, 48) = 16`——**改成 48
在本档位是空操作**。发现时任务已提交，索性留作**零假设对照**：
它应当与 base 落在同一分布上，用来标定这套同卡 ABBA 的噪声底。

（顺带：HCA 这里把上限写死 16，而 CSA 的 `decode_compressor_ratio4.py`
用 `COMMIT_WORKERS = 48`。B>16 的档位上 HCA 的 commit/pool/norm_rope_write
会比 CSA 少一半并行度，是一个独立于本轮的待办。）

### 98.16 ✗ `sc48` 确认轮翻转符号——第 98.12 节的"三项全负"作废

`abba27_sc48conf`（同卡 ABBA，CYCLES=5）对第 98.12 节的六样本结论复测：

| 轮次 | Δ平均 | Δmax | Δ极差 | Mann-Whitney U |
| --- | ---: | ---: | ---: | ---: |
| 第一轮 n=6（第 98.12 节） | −8.80 | −18.20 | −14.50 | 23.5/36 = **65%** |
| **确认轮 n=8/9** | **+11.50** | **+9.60** | −2.90 | 19/72 = **26%** |
| 两轮合并 n=14/15 | +2.98 | — | — | 90/210 = **43%** |

**两轮符号相反，合并后 U=43%（50% 即无效应）。`sc48` 没有可检出的效应。**

两轮各自都被判过"Δ 低于 base 极差，不可判定"（46.00 与 37.75），
我在第一轮却因为"逐名次差全部同号"往前推了一步，把它写成
"第一个三项全负的候选"。**六个样本的逐名次同号并不能替代显著性**——
n=6 时随机也有约 1/30 的概率出现单向排列，而 U=23.5 本来就没到阈值 31，
当时的记录里也写了这一点，却仍然在正文里用了肯定语气。

这与既有纪律一致：同一候选多轮符号翻转（`nz_cmpwgate` 四轮
−8.0/−13/−10.5/+2.75，`aicpu_thread_num=4` 两轮 −11.0/+5.0）说明
效应小于噪声，不是测量出错。

### 98.17 ★ 本 goal 下 SPMD 宽度 + `sync_start` 的完整结论

「尽可能用满核 + 叠加 sync_start」这条轴上测过的**全部**候选：

| 候选 | 改动 | Δ平均 | Δmax | 判定 |
| --- | --- | ---: | ---: | --- |
| `syncstart` | 121 处全开 sync_start | +65.96 | +57.50 | ✗ 有害 |
| `quant48` | quant 24→48（压流水 4→2） | +4.94 | −4.25 | ✗ |
| `quant48ss` | 同上 + sync_start | +2.29 | +3.50 | ✗ |
| `pool48` | softmax_pool 16→48 + sync_start | +13.61 | −4.75 | ✗ |
| `pool48ns` | softmax_pool 16→48 | +4.38 | −8.50 | ✗ 平均无收益 |
| `kvscore24` | kv_score 16→24（M 维 64→32） | +10.68 | +24.00 | ✗ 反向 |
| `krr8` | kv_rms_norm_rope 3→12 | +2.18 | +12.50 | ✗ U=44% |
| `warm24` | warm_kv 8→24 | −1.07 | +12.20 | ✗ U=44% |
| `sc48` | state_commit 16→48 | 两轮 −8.8/+11.5 | 翻转 | ✗ 无效应 |
| `sc48ss` | 同上 + sync_start | +16.29 | +5.25 | ✗ |
| `qproj24` | QPROJ_WORKERS 20→24（重验旧否决） | 在测 | | |

**十个候选无一为正。** 三条解释，按证据强度排序：

1. **每加一块要付约 0.27 μs 的 AICPU 派发**（第 98.10 节），而 span 由
   AICPU 而非计算核决定。填核的净收益 = 墙钟节省 − 0.27×Δ块数。
2. **与资源下限的 192.9 μs 差距，191.5 μs 落在"两侧都不满"的依赖阻塞窗口里**
   （第 98.9 节）。这类窗口里核空着不是因为块数不够，而是因为前序任务没完成，
   加块数改变不了它。
3. **`sync_start` 的代价正比于起点跨度**（第 98.13 节）：它要等最后一个核
   空出来，而跨度大恰恰说明核是陆续空出来的。

**下一步不应该继续沿这条轴找候选。** 真正的空间在那 191.5 μs 的依赖阻塞上，
要动的是任务之间的依赖与交错（例如 595–632 那 37 μs 是 8 组
proj_a→quant→proj_b 链的排空，AIC 只剩 10/24、AIV 只剩 4.5/48），
不是单个任务的块数。

### 98.18 ★★ 零假设对照：空操作量出 −8.48 μs，与 `sc48` 第一轮一模一样

`commit48cap` 把 `decode_compressor_ratio128.py` 的 `WORKERS` 16→48。
该常量在全文件只出现在 6 处 `pl.min(requests, WORKERS)`
（`softmax_pool` / `state_commit` / `norm_rope_write` 各两处），
B16 下 `requests = 16`，`min(16, 48) ≡ min(16, 16)`——**生成的 kernel 完全相同**。
建完变体才发现，索性留作零假设对照。

同卡 ABBA（CYCLES=3，每侧 6 样本）：

| 变体 | 样本 | 最小 | 平均 | 最大 | 极差 | Δ平均 | Δmax | U/36 | 逐名次同号 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :-: |
| base | 6 | 565.25 | 579.75 | 588.25 | 23.00 | — | — | — | — |
| **`commit48cap`（空操作）** | 6 | 551.50 | 571.25 | 593.00 | **41.50** | **−8.48** | +4.80 | **23 (64%)** | 否 |
| `qproj24` | 6 | 580.25 | 599.50 | 610.50 | 30.25 | +19.77 | +22.30 | **3 (8%)** | 是 |

对照第 98.12 节的 `sc48` 第一轮：**Δ平均 −8.80、U=23.5/36 (65%)**。

**一个什么都没改的变体，量出了与 `sc48` 第一轮数值上一模一样的"收益"。**
这为这套同卡 ABBA 标定了噪声底：

- **n=6 时，零效应变体的 |Δ平均| 可达 8.5 μs、U 可偏到 64%**；
- 因此 **|Δ| < 10 μs 且 U 在 [6, 30] 区间内的结果，一律等同于没有效应**；
- 极差也一样不可靠：空操作的极差 41.50，比 base 的 23.00 还大 18.5。

这条比第 68.1 节（≥6 样本才能定符号）和"Δ 要超过 base 极差"都更直接——
**它是实测出来的零点，不是推断。** 后续任何候选都应该和这个 −8.48 比，
而不是和 0 比。

⚠ 顺带作废我在第 98.12 节的推理："六个逐名次差全部同号"不能替代显著性。
空操作这次逐名次不同号纯属偶然，n=6 时单向排列的随机概率约 1/30，
一轮测三四个候选就会撞上。

### 98.19 ✓ `qproj24` 重验：旧结论成立，且这是本轮唯一达到显著性的结果

按「日志里的既有结论要用当前口径重测」重验 `QPROJ_WORKERS` 20→24
（旧记录只有七档加权均值 +9.33，没有 max/极差）：

**U = 3/36（n=6 的显著阈值是 ≤5），逐名次差全部同号，Δ平均 +19.77、Δmax +22.30。**

这是本 goal 下十一个候选里**唯一一个达到统计显著的**，而它是**负向**的：
**24 才是 AIC 满核，但 20 确实更快**，第 72 节「留余量」的判断复现成立。

机制在泳道里能看到：`qproj_matmul` 起于 190.3，而 `hca_kv_score_proj`
（16 个 AIC 块）要到 208.2 才结束。两者重叠的 18 μs 里
20+16 = 36 > 24，24 个 worker 会有 4 个等不到核、把整个任务拖成两波。
20 块换成 7 轮但恒为一波。

**这同时是第 98.8 节那张表第四行（"已为留派发余量而故意调低"）的实测背书。**

### 98.20 ✗ `state_commit` 的 `deps=[pool_tid]` 不是假依赖——是 WAR 顺序

查 206.5–285.5 那个 79 μs 阻塞窗口时注意到：`hca_state_commit` 的依赖是
`pool_tid`，而它的循环体只读 `kv_proj` / `score_proj` / `ape` / `state_slots` /
`positions` / `seq_lens`，**从不读 `pooled`**（读 `pooled` 的是
`norm_rope_write`）。按数据流看像是可以改成 `deps=[projection_tid]`，
让它提前约 18 μs 起跑。

**核对源码后否掉了。** `hca_softmax_pool` 在历史分支里读 `state`：

```python
values[history_row:history_row+1, :] = pl.slice(state, [1, POOL_COLS], [page, column])
scores[history_row:history_row+1, :] = pl.slice(state, [1, POOL_COLS], [page, column + HEAD_DIM])
```

而 `state_commit` 正是**写** `state` 的那个任务。所以 `deps=[pool_tid]` 是
真实的 **WAR（先读后写，Write After Read）顺序约束**，文件开头的 docstring 已经写明：
「先完成历史读取，再提交 Native state，避免滚动页覆盖」，
`softmax_pool` 体内也有「当前步的行在 UB 中覆盖历史读取；不先写 state，
不需要搬运窗口」。

要放松它，必须证明 `state_slots` 给当前 token 分配的页与 `state_table` 里
历史位置的页不重叠，而这取决于 cache 分配器的滚动策略——不在算子里可判定。
**结论：这条依赖不能动。**

记下来是为了避免以后再"发现"一次：**只看数据流（谁读谁的输出）会漏掉 WAR，
必须同时看谁写了对方读的缓冲。**

### 98.21 ⚠ 无卡"编译校验"抓不到 `@pl.jit` 的 trace 错误

`nodummy` 第一次提交时编译失败：
`UnsupportedFeatureError('Unsupported expression type: ListComp')`。
原因是我把 `return partials, act_scale_dq, [proj_b_tids[i] for i in range(O_GROUPS)]`
写进了 return——**PyPTO 允许参数位置的列表推导**（原代码
`task_dummy(deps=[proj_b_tids[i] for i in range(O_GROUPS)])` 就是），
**但不允许 return 里的**。改法是返回 `proj_b_tids` 数组本身，
让每个消费者在 `deps=[...]` 参数位置展开。

随后我写了一个无卡校验脚本（把变体包覆盖进 `PYTHONPATH` 再 import
`HCAOperators` 触发注册），两个变体都报 OK。**但拿故意写坏的副本反证，
它同样报 OK**——注册只走到算子登记，`@pl.jit` 的 trace 要等首次带真实 shape
调用才发生，所以这类错误无卡查不出来。

**这条校验路径作废，不要再用。** 正确做法是直接提交 A/B：
`run_hca_ab_same_card.sh` 会把失败的 pass 如实记成
`None  UnsupportedFeatureError(...)`，只废掉一个 pass，不影响其余变体。

### 98.22 O 投影尾部的两个依赖重排（在测）

用户 2026-09-30 批准「改依赖与交错」。泳道（128K/B16）：O 投影段
504.2–668.1 = 163.8 μs，AIC 下限 101.9，差 **61.9 μs（占总空隙的三分之一）**。

| 组成 | 核·μs | 单元 | 下限 |
| --- | ---: | --- | ---: |
| `proj_a_mm` ×8 组 | 1 723.2 | AIC | 71.8 |
| `proj_b_mm` ×8 组 | 721.7 | AIC | 30.1 |
| `quant` ×8 组 | 949.3 | AIV | 19.8 |
| `hca_oproj_hc_post` | 1 662.5 | AIV | 34.6 |

差距的构成：组阶段排空约 15 μs + `task_dummy` 那一跳 6.8 μs +
`hca_oproj_hc_post` 独占 39.4 μs（期间 **AIC 完全空转**）。
而组阶段（504–622）AIV 只有 5.5~16/48 在跑——**post 的 AIV 工作量
本来完全装得进那个空当**。

两个候选：

1. **`nodummy`**：删掉 `parts_ready = pl.system.task_dummy(deps=[8 个 proj_b])`，
   让 `hca_oproj_hc_post` 直接依赖 8 个 tid。最后一个 proj_b 结束于 621.8，
   post 到 628.6 才起跑，中间 6.8 μs 就是这一跳 + 48 块派发。
   本仓库已经因为同样理由删过一次这种 dummy
   （`decode_compressor_ratio128.py` 的注释）。不改数值、不改块数、不加任务。
2. **`postsplit`**：把 post 里 `for group in pl.pipeline(O_GROUPS)` 的求和拆成
   g=0–3 / g=4–7。前半段 `hca_oproj_acc_lo` 只依赖前 4 个 proj_b
   （分别结束于 592.7 / 596.7 / 599.5 / 601.2，比 628.6 早约 27 μs），
   那段 AIV 占用仅 1.4~9.6/48。累加次序仍是 g=0→7，FP32 经 GM 往返不丢精度。

⚠ 环境记录：本轮无卡校验的日志显示当前 CANN 为 **9.2.0-beta.2**
（`/data/pyptouser/yejia/vllm-cann92-main/env/Ascend/cann-9.2.0-beta.2`）。
公共入口会被其他 session 改动，跨轮的绝对数字不可直接比较，
同卡 ABBA 的轮内 Δ 不受影响。

## 99. 接手后先做residual复用，核内与调度交替推进（2026-09-30）

目标仍为比Native快20%以上；不因旧试验没有收益而缩小范围或宣布不可达。
旧第91节的同步计数推论已被第92节反例推翻，HANDOFF此前仍引用它，现已修正。
第98.21节仅证明import/register不触发完整trace，不能外推为CPU不能校验；
本轮对基线和候选显式调用两根_get_dep_graph()、test根compile(RunConfig(platform="a2a3"))并load，均通过，未占NPU。

发现HCA的o_proj_hc_post按out_h展开4次，每次又按in_h加载/转换4份residual，共16次。
最新本地ops-transformer的mhc_post/arch22含USE_PERMANENT_X路径：一次CopyInX加载全部HC分量，
ComputeCopyOutAllX一次转FP32，在多个输出间复用。它是可选分支，不声称Native当前形状一定选择了该分支。
本轮采用同一数据复用思路：四份8×512 FP32 residual保留在UB，各输出仍按0/1/2/3加法顺序计算。
保持token/列分工、AIC任务图、post/comb权重、BF16舍入、slot和cache布局；不叠加sync或改依赖。
代价是UB驻留增加，完整CPU编译通过；生成核可见四次加载与四次转换在out_h展开前，尚不宣称性能收益。

实验基底c6792787，整包复制到.cache/hca-residual-c6792787-20260930/base与reuse后只读冻结，
不修改另两个在跑实验的包或生产代码。task_20260930_011026_17930231765经auto队列运行，
先128K/B16同卡ABBA四个pass，再各采独立泳道；性能后一次比对已保存完整输出/cache/state。
不继承每候选十个进程的重测规模；小效应若不可辨认先看核内收益及关键路径，再决定是否有必要补测。
旧摘要的min/max是pass中位数的范围，本轮另报全部单次span的min/max/mean，避免低估长尾；
图外事件含主机间隔，继续单列，不与纯设备span混算。原分段结果保留每重放kernel数供核对。
[hca_residual_reuse_20260930](hca_residual_reuse_20260930/README.md)。

### 99.1 两档完成：保留核内复用，不宣称整层收益

补充8K/B24任务task_20260930_011655_2614578312已完成。两档全部输出及完整cache/state
在各自ABBA中与首份base逐bit一致，eager/graph与保护区检查通过。无Native或整模型新增验收。

以下为全部18次单次设备重放的min/max/mean，单位μs，非pass中位数的极值：

| 档位 | base min/max/mean | reuse min/max/mean | mean变化 |
| --- | --- | --- | ---: |
| 128K/B16 | 565.50 / 617.50 / 583.31 | 568.00 / 636.00 / 599.17 | +2.72% |
| 8K/B24 | 604.50 / 646.50 / 625.18 | 603.25 / 662.50 / 620.08 | −0.82% |

独立DFX的post任务，长档48 worker核内min/max/mean：32.28/38.80/35.26→22.30/28.96/26.09；
短档72 worker：18.72/42.44/31.93→17.70/38.60/29.74。两侧形状相同；
post从首开到末结束长档39.00→32.22、短档62.30→58.16。
按“核内有收益就保留，调度继续优化”的要求保留residual复用。
这不是整层性能通过：长档变慢、两档max升高，不能只报post的26%当成HCA提速。
长档未改的attention AIC mean155.03→169.26、O-A mean24.03→29.81，
其启动关系也发生变化；本轮证据不足以给这些变化指定唯一原因。后续检查多task联动。

顺带读取接手时已排队的三个历史调度实验，不新增测试：group候选mean整体变慢；
swapnrw的单次全样本base min/max/mean=557.25/668.25/584.94，
候选547.00/649.75/591.64（mean变慢6.70，max降低18.50，保留为待分析线索）；
postsplit为554.50/631.50/592.67→569.25/658.00/620.89，暂不接入。
详见同实验目录inherited_schedule_results.json；不重复旧的pass中位数max口径。

## 100. 按实际重叠关系检查Q投影与反量化的联动（2026-09-30）

用户再次要求注意多task联动。静态核数之和不能代替真实泳道：Q投影与KV/Compressor
共享AIC，反量化与gather/state共享AIV；单组变快也可能使另一组延后，或减少attention重叠。
本轮保持核数、分块、核内流水与真实依赖，只比较base、Q投影sync_start、
Q投影+反量化联动sync_start三方。长档首轮同卡正反各一次，共六个进程，
再各采独立泳道；不机械全开sync_start，也不从旧失败候选推断所有组合都无效。
基底为第99节的residual复用；所有包只读冻结、CPU完整编译通过后上卡。
任务task_20260930_012821_4472725985，证据入口
[hca_qchain_schedule_20260930](hca_qchain_schedule_20260930/README.md)。
报告全部单次min/max/mean，以及任务开始跨度、完成时间和链路重叠；
性能之后一次比较保存状态。未修改state_commit的WAR依赖或公共运行时。

### 100.1 完成：Q组变整齐，但联动同步将等待转移到反量化

六进程与三份DFX全部完成，状态与保护区通过；每侧18次设备span：

| 变体 | min | max | mean |
| --- | ---: | ---: | ---: |
| base | 562.00 | 645.00 | 582.60 |
| syncq | 568.25 | 643.25 | 590.25 |
| syncboth | 588.75 | 649.00 | 607.90 |

本轮不接入两项同步策略，不为负收益补跑8K。
独立DFX显示Q启动跨度63.78→0.62/2.16 μs，确实更整齐；
但Q结束到反量化首次启动间隔6.32→14.78/45.70 μs。
syncq的gather在246.88–315.58、反量化在256.58–318.68，两者重叠；
syncboth的gather在240.32–284.68、反量化在290.46–343.70，发生串行化。
attention的首次启动base/syncq/syncboth为338.96/320.20/349.12。
因此只看单组跨度下降会误判，必须检查生产者、并行旁路和消费者一起的时序。
单张syncq泳道虽更快，独立ABBA均值仍更慢；不将单窗口机制线索写成稳定整层收益。
完整输出/cache/state逐bit一致；证据见本节链接的result_h131072_b16.json。

## 101. 恢复核内优化：完整有效压缩块省mask的组合未受益（2026-09-30）

参考最新ops-transformer的SparseAttnSharedkv A3 `DealBmm1ResBaseBlock`：
缩放score后按实际有效列数进入softmax。PTO候选由gather发布实际搬入行数，
只有当前query可见完整K128且所有页真实有效时省mask/bias/fillpad/乘mask，
尾块与无效页继续原分支，不改变分块max、BF16概率和running归约。
新增计数按64字节行距隔离，避免并发scalar write共享cache line；
CPU先修正嵌套yield与行距，再编译通过上卡，没有用设备试错这两项问题。

task_20260930_013917_58586026298完成，128K/B16同卡ABBA，每侧18次设备span：

| 实现 | min | max | mean |
| --- | ---: | ---: | ---: |
| base | 557.25 | 624.25 | 582.63 |
| fullmask | 578.75 | 645.25 | 605.39 |

独立DFX的attention AIC mean148.04→166.96、AIV152.03→171.05，
AIV max155.62→174.18，整组跨度155.80→174.26 μs。
gather mean26.33→21.04而组跨度34.80→36.48，不能把局部均值下降当成整组提速。
没有目标attention核内收益，不接入、不补短档，不扩大功能测试。
已有完整输出/cache/state跨版本逐bit一致、自身图重放和保护区通过；仅代表正常长档，
不声称无效页/边界/Native模型已验收。额外标量读取与分支是待查成本，不能唯一归因。
证据：[hca_fullmask_20260930](hca_fullmask_20260930/README.md)。

下一项可用实现参考已经核对：CSA第430–432节的O-B AL1-full/N-first二版，
保留K128 L0流水，在ROW≤96时把一次激活载入复用给两个N256输出。
HCA当前尚未吸收；只能按函数移植评估，不能把CSA的其他O投影调度/后处理一起复制。
仍按核内与调度交替推进；本阶段默认只新增第99节residual复用，20%领先目标尚未达成。

## 102. O-B激活常驻L1：保留核内收益，整段没有改善（2026-09-30）

从CSA提交5c66a59b只提取NZ O-B函数，参考ops-nn AL1-full/N-first：ROW≤96时，
一次加载完整K1024激活，复用给两个N256输出；权重K256 stage2、L0 K128双缓冲不变。
ROW128、ND、任务图及算术不变。整包冻结，基线和候选CPU显式编译后上卡。

同卡ABBA每侧18次真实单次设备span，单位μs：

| 档位 | base min/max/mean | al1 min/max/mean |
| --- | --- | --- |
| 128K/B16 | 543.00 / 627.50 / 571.54 | 556.25 / 629.50 / 573.69 |
| 8K/B16 | 423.25 / 491.50 / 439.15 | 426.25 / 508.25 / 446.24 |

整段mean分别+0.38%/+1.61%，max也增加，不宣称性能验收通过。
独立DFX中O-B核内mean为11.834→9.542（−19.37%）和8.358→7.793（−6.76%），
max为17.36→11.72和11.78→9.04；组跨度长档67.48→60.94，短档48.98→50.14。
长档未改的O-A也从27.56→21.60，短档则15.23→15.53，不能把联动变化全部归因为激活复用。
按“核内有收益就保留”的既定取舍接入O-B，后续继续优化完整路径及最大值；这不是整层收益声明。

两档完整输出和三组cache/state逐bit一致、图重放与保护区通过；另补128K/B4覆盖ROW32、
24个有效token及尾部，也逐bit一致。没有扩大到七档或整模型；既有Native基线未重测。
任务、所有span、两侧DFX与源码差异见
[hca_ob_activation_20260930](hca_ob_activation_20260930/README.md)。
本轮format.sh ci因环境缺少pre-commit未运行成功；修改文件另做Ruff、shell语法与diff检查，
不把这些定向检查冒充完整CI通过。

## 103. 同进程交错筛选：缩短实验准备，但小效应仍不可辨认（2026-09-30）

两个PTO候选没有Native静态包互相污染的问题。本轮把冻结包挂在两个独立Python别名下，
注册不同Torch op，各自npugraph_ex捕获，复用权重与Native fixture，按ABBA交错10次/侧。
先验证同源码A/A：min/max/mean为579.50/609.50/590.58与585.00/623.25/595.40，
同源码也有4.825μs均值差；因此只用它快速筛选，不用来给微小收益背书或替代服务入口验收。
O-B同进程对照mean589.38→595.45，也没有整段收益。完整输出/cache/state逐bit一致，保护区通过。
profiler缺失ACL到NPU flow连线，不能做CPU→NPU因果图；原始设备CSV的20段各含两执行核，分段检查通过。
首次运行遗漏Native metadata算子初始化，已修复；失败运行无性能数据。
脚本、校准结果和限制见[hca_pair_screen_20260930](hca_pair_screen_20260930/README.md)。

## 104. Q提前解析未把局部提前传递到完整路径（2026-09-30）

在bef9f7fa基线上，只对NZ Q投影补allow_early_resolve=True；保持20核、依赖和流水。
128K/B16同卡ABBA每侧18次span，base min/max/mean为571.00/671.00/597.46，
候选561.75/668.75/596.83。0.625μs不足以证明收益，暂不接入，不扩大短档测试。
两次base进程均值585.61/609.31、候选582.50/611.17，有明显漂移。

独立DFX的Q结束275.84→229.08，但反量化跨度48.24→86.16，压缩gather跨度36.98→60.20；
attention AIV开始334.94→330.28、结束504.34→523.00。Q和Compressor竞争AIC后，
向量侧的并发关系也改变了，不能用Q单组缩短45.6μs等价整段收益。
精度与保护区通过。完整证据：[hca_qearly_20260930](hca_qearly_20260930/README.md)。

## 105. 长档softmax统计留在UB，保留核内收益并继续处理长尾（2026-09-30）

最新本地ops-transformer 28f40354的SWAVectorBlock将softmaxMax/Sum保留在UB。
PTO当前maxima/totals却由同一AIV写GM、两个tick后再读回；它们不参与跨核交接。
本轮改成每AIV独立的三槽UB环，scores/probs/values、FFTS次数、max/量化/归约顺序均不变。

v1虽然显式CPU编译通过，设备跨版本逐bit失败，未计时。生成代码显示动态tile.slice
经reshape后绑定环首地址，写入端却保留了动态槽偏移。v2用显式Vec extract生成动态TEXTRACT，
完整输出/cache/state恢复逐bit一致。不修改公共PyPTO/Simpler/PTOAS/PTO-ISA；
局部证据见[COMPILER_SLICE_NOTE.md](hca_softmax_ring_20260930/COMPILER_SLICE_NOTE.md)。

128K/B16独立DFX：attention AIV核内min/max/mean从186.78/192.60/189.74变为
174.74/181.62/177.84，mean下降6.27%、max下降10.98μs；AIC mean185.64→173.76。
同进程筛选则600.53→610.23，不能只报核内收益当作整层收益。
未改的其他task也变化，gather组跨度34.86→69.32，再次表明需观察多task联动。

与已提交O-B版本组合，128K/B16和16384/B4都进入长分支，完整状态及图重放/保护区通过。
后者页表[4,6]、128行有效压缩历史，覆盖只有压缩块加raw的短循环排空。
同进程组合筛选577.20→596.85，因此又用生产服务入口同卡ABBA核对，每侧18次真实span：

| 实现 | min | max | mean |
| --- | ---: | ---: | ---: |
| 已接入O-B的base | 585.75 | 638.50 | 600.86 |
| O-B + UB环 | 561.50 | 655.50 | 592.21 |

本窗口mean减少8.65μs（1.44%），max增加17μs。两种筛选方向不同、进程间漂移明显，
稳定整层收益和长尾改善都未证明。依既定规则保留核内收益，短档路径不改；不扩大负/不明收益的七档或模型测试。
已有Native基线不重标为当前版本成绩，领先20%的目标仍未达到。
完整脚本、数据、任务和泳道：[hca_softmax_ring_20260930](hca_softmax_ring_20260930/README.md)。

新增脚本Ruff、shell语法及diff检查通过；attention文件原有I001/E501检查项由61项变为60项，
没有新增诊断，未做整文件无关格式化。完整format.sh ci仍受缺少pre-commit限制。

## 106. 下一轮：按Native跨query延续流水，而非重复排空（2026-09-30源码审查）

本地ops-transformer 28f40354提供了两个尚未吸收的具体结构参考，尚无新性能结论：

1. `sparse_attn_sharedkv_swa_kernel.h`约790行只在当前核的最后一项设置`extraLoop=PRELOAD_NUM`，
   用持续递增的gloop驱动PreloadPipeline，跨query延续QK/PV流水。
   当前PTO `_long_sparse_attn_hca_tp1`对每个token执行`work_count + QK_PRE_LAUNCH`，每次重新启动/排空。
   128K/B16每worker有4个token，单从循环结构看可避免3次重复排空；这不是已测的时间收益。
2. `sparse_attn_sharedkv_tiling.cpp`约1538行，CFA/SWA的mBaseSize为`(256/gSize)*gSize`，
   H=64时覆盖最多4个query，PTO仍按单query的M64处理。不要把SCFA的`mBaseSize=gSize`
   当成C128 HCA的唯一参考；模式选择依据cmpSparseIndices是否存在。
   本地源码策略不等于已经证明安装包为本形状选择了相同内部分块，后续需保留这一边界。

先做跨query延续流水，再独立评估多query拼块，避免一次叠加两类改动。
跨query方案必须保留：压缩先于raw的既有数值顺序；每query的sink/位置/mask；
三槽KV/probs/values的生命周期；PV完成时对正确query做inverse RoPE与输出；
seq_lens=0及不同请求work_count的尾部。QK已进入下一query时，PV可能仍归约上一query，
不能直接沿用当前token的归约状态或频率。仍先冻结包CPU编译、代表长档筛选，数值中性检查后再补受影响边界。

## 107. 跨query流水已保留；分别核对核内收益和多task联动（2026-09-30）

落实第106节第一项。基线bb4c2831，CANN9.2、NZ2、atomic0、deterministic0，
正式第3层权重；私有完整包CPU依赖图解析、设备编译和加载通过后才排队。
只改长档attention：KV三槽跨query保留，QK/PV用递增work_offset，worker最后一条query才排空。
同AIV的UB描述符环记录token/局部块号/剩余块数，使延迟PV的归约复位、RoPE和输出保持正确。
压缩先于raw、BF16概率转换点及归约算术不变；任务数、依赖和24组MIX核不变。

逐次纯设备span，单位μs。服务入口ABBA每侧18次，同进程ABBA每侧10次：

| 口径 | base min / max / mean | stream min / max / mean |
| --- | --- | --- |
| 128K/B16 同进程 | 586.50 / 631.75 / 608.45 | 574.50 / 604.50 / 584.33 |
| 128K/B16 服务入口 | 587.00 / 648.75 / 605.71 | 558.50 / 634.75 / 583.90 |
| 8K/B24 同进程 | 622.25 / 669.75 / 637.35 | 633.00 / 684.75 / 644.73 |

长档服务入口mean减少21.81μs（3.60%）、max减少14μs，两种测法方向一致。
短档路径没改，本窗口仍记录+7.38μs波动，不能直接宣称短档零回退。
同进程长短档mean相对变化按8:2加权为−2.94%，本轮保留；不把这组PTO前后对照当作新Native七档成绩。

### 107.1 多task联动：核内变快不等于所有时间差都归该task

独立DFX的attention AIV min/max/mean为157.90/163.86/161.74→146.80/150.94/149.03，
核内mean下降7.86%。但attention起跑353.18→359.16，晚了5.98μs，整组结束517.46→510.50，
实际只提前6.96μs。上游Q反量化结束347.78→352.92也更晚。
未修改的O-B核内mean10.85→9.26，mHC post35.25→25.89；cmp gather反而22.31→30.13。
因此整段21.81μs改善包含多个task的运行变化，不能全部归给attention自身的12.71μs。
泳道的并发/等待只用于定位；没有单独证明L2或某项资源竞争的因果。

后续继续按用户补充要求：识别生产者、消费者和并发任务，必要时做联动候选，
同时报告核内min/max/mean与整段min/max/mean；清晰区分“核内收益保留”和“整段收益已证实”。
不一口气给所有SPMD加满核和sync_start，也不因单task先结束就判有效。

### 107.2 有限且针对性的边界检查

长档、8K守护及服务入口两侧完整输出/cache/state逐bit一致，eager/图重放、保护区通过。
16507/B6覆盖压缩块128→129行的边界、24组worker分36条query的不均分配，
同址图6→5→1→6通过；16384/B2覆盖无query worker与短循环，2→1→2通过。
边界cycles=1只作功能依据，不宣称性能。

B2第一次在满档2/2报错，旧base同样复现：三份完整状态、输出、保护区均通过，
仅因本步无新压缩token，测试把空cos/sin/slots切片判FAIL。
已修正HCA padding检查：零有效行时比较完整负slot缓冲，不消费的cos/sin不做空张量比较。
只补跑受影响B2后通过；筛选脚本新增padding诊断和失败报告，避免只有异常堆栈而无逐项证据。

所有脚本、任务号、摘要和两侧泳道见[hca_query_stream_20260930](hca_query_stream_20260930/README.md)。
生产代码与已验证私有包仅注释/换行不同，AST一致；新增脚本Ruff、shell/diff检查通过，
attention历史60条I001/E501未增加。完整format.sh ci仍因环境缺pre-commit未通过。
当前领先Native20%的目标尚未达成，后续继续核内与调度交替推进，优先核对多query复用和其调度影响。

## 108. Q生产者/消费者提前解析联动：均值未获益，保留削峰线索（2026-09-30）

基线a7a9f314；CANN9.2、NZ2、atomic0、det0、正式第3层权重。
两版分别关闭Q反量化allow_early_resolve，以及同时开启Q投影allow_early_resolve。
保持核数、tiling、任务依赖与算术，私有包CPU显式编译通过后同卡ABBA。
128K/B16每侧10次真实span，μs，min/max/mean：

- late_dq：base 547.25/592.50/562.60 → 候选561.00/600.25/582.85。
- early_q_late_dq：base 587.50/634.00/602.73 → 候选583.25/622.75/603.90。

单关消费者退化；联动mean基本持平，max减少11.25μs，按要求保留削峰线索而非一概否定。
尚不修改生产路径；不把两个独立轮次的绝对值混为候选排序。
四类完整输出/cache/state逐bit、自身图重放与保护区通过。
[脚本、两侧报告与边界](hca_qresolve_20260930/README.md)。无新整机或七档验收。

## 109. Q激活常驻L1的三个核内候选：生成流水与多task时序都必须看（2026-09-30）

参考本地ops-nn19614968的MatmulBaseKernelAL1FullLoad，不声称安装Native本形状使用此模板。
基线与配置同108，先整K1024 A常驻，再四个K256独立A常驻，最后叠加N128外层stage2。
所有版本保持INT8整数累加顺序、20workers、task依赖；仅NZ分支修改。
完整CPU解析/编译/加载及生成内存分配检查通过。每候选128K/B16同卡ABBA各10次，μs：

| 版本 | base min/max/mean | 候选 min/max/mean |
| --- | --- | --- |
| 整K1024常驻 | 543.00/591.25/564.28 | 566.75/598.50/577.18 |
| 四独立K256常驻 | 566.75/606.25/579.65 | 582.50/603.25/593.20 |
| 四块+N128 stage2 | 555.00/628.00/567.10 | 562.00/593.25/570.93 |

整块版Left由Null/TMOV变Normal/TEXTRACT，Q核内mean 59.057→64.055μs，故不接入。
纠正最初冻结注释：原版本和候选L0均为K256，不是K128；prepare脚本已更正，不篡改已测包。
四独立块恢复TMOV，Q核内mean 61.03→59.60，但组跨度73.72→121.60，反量化更晚。
保留局部复用实现和证据，尚不能称稳定核内/整段收益。
N128实际Left/Right双槽各32KiB，Acc只有一个64KiB槽，Mat384KiB；stage2不等于L0C双缓冲。
其Q核内mean 61.03→74.49，Attention/post却同时变快，不能把整段max降低归因于Q更快。
三版完整输出/cache/state逐bit、自身图重放与保护区通过；不扩展未获益候选的七档或整机测试。
[完整结果、任务组起止、源码与泳道路径](hca_q_activation_20260930/README.md)。

下一步已定位Q RoPE逐行Gather的重复标量循环，尝试整块flat Gather；同时重取当前版本与
Native static/SuperKernel的长档同卡ABBA，防止持续只报PTO前后优化而失去20%目标分母。

## 110. 重取Native static/SuperKernel基线：当前PTO仍慢12.97%（2026-09-30）

task_20260930_032530_227722523167在同卡1完成Native/PTO/PTO/Native四pass，
每侧18次实际设备span。a7a9f314冻结生产包，CANN9.2、NZ2、atomic0、det0、正式第3层权重。
两侧npugraph_ex dynamic=False、inplace/static开启；Native的SK实际静态编译/装包均核实。
PTO保持SK关闭：上一任务032122_216888230045在PTO SK1图优化报107017无效funcHandle，
故完整ABBA重跑，没有混入失败轮的Native计时。未修改公共工具链或Native执行流程。

| 实现 | min | max | mean | P50 |
| --- | ---: | ---: | ---: | ---: |
| Native static + SK | 533.25 | 562.25 | 544.35 | 541.50 |
| PTO a7a9f314 | 590.75 | 646.50 | 614.94 | 609.50 |

单位μs；PTO mean慢12.97%、P50慢12.56%。本档20%目标为P50≤433.20，还需减少176.30μs。
前面PTO自身优化不等同达到Native目标，旧版Native分母不能挪用；本档也不代替新七档/整机验收。
保护区和有限值通过，PTO自身图重放精确；Native编译图与eager零容差比较的输出/SWA有差异，
两pass均为15814/9个元素，max_abs分别0.015625/0.0078125；compressed/state精确。
保留该诊断，不宣称Native/PTO跨实现精度或token已通过，也未把小浮点差异直接定为功能失败。
[完整配置、原始样本与复现](hca_native_anchor_20260930/README.md)。

## 111. Q RoPE整块Gather已保留：核内下降20.4%，整段等待仍需优化（2026-09-30）

生成代码显示原axis Gather每个head对8行逐行循环，包含8次Gather与临时行搬运。
改为给index加row×64后一次flat Gather，行偏移每head组准备一次；
核数、stage2 head流水、依赖、RMS/量化/舍入顺序保持不变，尾块代码不改。
参考ops-transformer28f40354 interleave_rope_b11d按多行批量处理向量操作的思路；
不声称Native安装包用了相同模板。pypto-lib2164563仍用axis Gather，此处有意改表达来减少标量循环。
不修改PyPTO/Simpler等共享仓；显式CPU编译/加载通过，生成代码已消除逐行gather循环。

同进程ABBA各10次，μs，min/max/mean：

| 档位 | base a7a9f314 | flat Gather |
| --- | --- | --- |
| 128K/B16 | 587.50/616.50/596.75 | 578.75/624.75/597.23 |
| 8K/B24 | 617.00/671.25/637.78 | 621.00/638.25/631.03 |

长档mean持平、max+8.25；短档mean−6.75、max−33。8:2加权mean为−0.148%，不能称稳定整段提速。
独立DFX的Q反量化核内min/max/mean 40.78/48.60/44.55→34.52/36.52/35.48，
mean下降20.4%、max下降12.08μs，按用户要求保留核内收益。
但该组跨度49.24→64.74变宽，Attention AIV核内mean161.49→176.95；
Q投影启动分布与gather重叠也同时改变，整段效果不能简单等于本task减少的9.07μs。

长短档与16507/B3的四类完整输出/cache/state逐bit，自身图重放和保护区通过。
B3包含完整8行块与2行尾，同图3→2→1→3通过；其cycles=1只作为功能证据。
生产helper与已测私有包AST一致。第110节Native对照是改动前基线，未冒称本次新版本Native/整机验收。
[完整实现、联动表、任务与泳道](hca_q_flat_gather_20260930/README.md)。
新增脚本Ruff、shell/diff通过；全仓format.sh ci受缺pre-commit限制。后续交替转入调度，
优先针对变快的Q反量化与并发gather的组等待，再继续Native多query复用方向；目标保持20%以上。

## 112. Flat Gather之后的同步联动：单组收拢仍会把等待转给旁路（2026-09-30）

基线ad0e6bbe、CANN9.2/NZ2/atomic0/det0/正式第3层权重。两个私有包保持算术、
48个Q反量化worker、原head流水及依赖，只改sync_start：单独Q反量化，以及Q反量化+长档压缩gather。
CPU显式编译/load通过，task_20260930_034114_25845992477同卡ABBA每侧10次，单位μs：

| 候选 | 自己的base min/max/mean | 候选 min/max/mean |
| --- | --- | --- |
| dq_sync | 574.50/628.50/594.10 | 584.00/616.75/596.48 |
| both_sync | 554.50/600.25/572.08 | 580.75/598.00/587.33 |

前者mean持平、样本max下降11.75，保留削峰线索；后者mean增加15.25，不接入。
完整输出/cache/state跨版本逐bit、自身图重放与保护区通过，没有扩大为短档/七档/整机验收。
独立DFX中dq_sync的Q反量化跨度54.38→46.52，但gather跨度37.70→96.52，
Attention启动309.96→341.94；任务链整体没有因单组变整齐就变快。
此窗口没有证明唯一竞争根因；不能将一次低max写成稳定削峰。
[全部数据、任务与泳道](hca_flat_sync_20260930/README.md)。

## 113. 双query共享压缩KV：三种容量/流水方案均未获益（2026-09-30）

参考ops-transformer28f40354的CFA/SWA多query M组织及arch22分块FP32累积GM读写；
没有把本地模板当成安装Native本档实际tiling的证明。ad0e6bbe冻结基线，长S6相邻两query
压缩部分M128共用KV、raw部分仍各自M64窗口，每query块顺序、BF16概率、sink与RoPE保持。
24个MIX组及依赖不变，未改短档。先两槽+GM累积，再前半UB后半GM，最后恢复三槽预发射。

全UB版需要240384字节，超过当前编译配置Vec188416，未提交NPU；半UB初版因尾部使用
使整片继续存活，同样超限。改成最终按H16重新加载后编译Vec174848；三槽版Q/P复用L1，
Mat恰为524288，代价是每块重新加载Q。没有通过修改工具链容量绕过限制。

同进程ABBA各10次，CANN9.2/NZ2/atomic0/det0/正式第3层，128K/B16，单位μs：

| 候选 | 自己的base min/max/mean | 候选 min/max/mean |
| --- | --- | --- |
| 两槽+GM累积 | 583.00/627.00/596.40 | 599.50/631.50/613.45 |
| 两槽+半UB累积 | 558.50/602.00/573.25 | 563.00/596.25/578.95 |
| 三槽+GM累积 | 577.50/611.50/590.08 | 604.25/631.75/618.28 |

完整输出/cache/state跨版本逐bit、自身图重放与保护区均通过；未扩大为七档/整机验收。
DFX两槽GM的Attention AIV mean164.73→182.73，另轮半UB/三槽为161.66→191.23/202.89。
AIC亦变慢。两精化候选attention启动347.32→318.00/314.44，但结束512.48→513.22/524.72；
不能将上游调度提前误当核内提速。半UB样本max−5.75不足以证明稳定削峰，三个版本均不接入。
生产仍ad0e6bbe；保留源码与容量边界，不泛化为所有多query方向都失败。
[源码、统计与全部任务泳道](hca_pair_query_20260930/README.md)。

下一项转mHC pre混合值常驻UB。现有DFX中mix+RMS核内约38μs，上游widen、linear与
并发Sinkhorn/预热共同决定前段时序；独立Native SK的HcPre+RmsNorm为48.53μs，
分组/采样不同，仅用于定位方向，不直接相减声称可优化空间。

## 114. mHC pre混合值UB常驻与原始BF16读取：减少流量未转成核内收益（2026-09-30）

参考最新ops-transformer28f40354的ProcessY及本仓Native直接读原始输入再转FP32的方式，
在ad0e6bbe冻结包上逐项对照：混合值UB常驻、只读原始BF16、两者叠加。线性投影仍FP32，
12个mix worker、D256 stage2、固定归约次序和BF16舍入不变。pypto-lib2164563仍存取GM，
本轮改变存储和读取方式，未改其数值策略。整段采用正式第3层权重/CANN9.2/NZ2/atomic0/det0。

初始横向UB组装不满足A3 TMOV约束，改成16个8×256连续块；显式Tile高精度rsqrt用scratch。
CPU解析/编译/load通过，三个候选Vec静态分配边界131360/74016/139552，未改工具链容量。
128K/B16同进程ABBA每侧10次，min/max/mean，单位μs：

| 候选 | 自己的base | 候选 |
| --- | --- | --- |
| UB混合值 | 551.50/588.25/571.48 | 574.25/592.75/582.73 |
| BF16输入 | 576.50/742.25/606.90 | 571.50/606.75/588.40 |
| UB+BF16输入 | 550.50/582.00/561.88 | 558.75/591.00/572.65 |

BF16轮mean/max下降，但base有一次742.25慢点且P50为588.38→588.75，不声称稳定收益，
原始慢点完整保留。独立DFX mix核内mean38.47→41.03/40.23/41.93，均未改善，暂不接入。
Sinkhorn与预热实际重叠、Q起跑/跨度也同时变化，没有只看mix或把静态指令数当成唯一成因。
四类完整输出/cache/state跨版本逐bit、自身图重放/保护区通过。未新做短档、七档或模型验收。
[全部结果、源码、任务与泳道](hca_pre_ub_20260930/README.md)。

接着细分mix为有效4行：物理4行的FP32列向量只有16B，故保持8行物理盒、有效4行，
门控直接留UB并用1×8 Gather后reshape，避免窄转置/Gather对齐问题。
所有读写限制有效行，去掉多worker会竞争的旧单尾块scratch。B16为24个mix，Sinkhorn仍12个，
不盲目把两组都扩到48。另版D512加载/混合/归一化，统计仍按两个D256原顺序累加。
两版CPU编译/load通过，task_20260930_043138_334282719249筛选已完成；后续取舍见第115节。

## 115. mHC mix有效M4与D512已保留：核内mean减少29.93%，整段联动待继续（2026-09-30）

物理盒保持M8，24个worker各处理4个有效token（128K/B16）。门控直接留UB，
1×8扁平Gather后reshape避免窄转置限制；所有加载/写回明确有效行，不使用共享单尾块暂存。
保留原始BF16输入、FP32混合与BF16舍入，按D512搬运/混合/归一化；
RMS每个D512仍拆两个D256，依0、256、512…原顺序累加，stage2与依赖不变。
Sinkhorn仍按8行分工，没有把并发任务一并扩大到48核。

M4/D256长档整段mean572.23→575.88，mix核内mean37.885→31.82。
D512进一步降低核内：min/max/mean为37.06/38.62/37.885→25.16/28.80/26.548，mean−29.93%。
独立DFX组跨度38.64→29.08，QA启动110.06→102.78；Sinkhorn结束99.40→103.26，
其他投影/attention也有联动变化，不能简单相加核内收益当整段收益。

同进程ABBA每侧10次真实span，CANN9.2/NZ2/atomic0/det0/正式第3层，单位μs：

| 档位 | base min/max/mean | M4/D512 min/max/mean |
| --- | --- | --- |
| 128K/B16 | 548.50/597.25/569.23 | 556.00/589.25/571.18 |
| 8K/B24 | 592.00/650.00/613.40 | 602.50/631.75/615.88 |

长短mean分别+1.95/+2.475，8:2相对mean约+0.355%；样本max−8/−18.25。
尚无稳定整段收益结论。按用户“有incore收益就保留”的规则接入D512版，继续优化调度。
长短四类完整输出/cache/state跨版本逐bit、自身图重放及保护区精确。
16507/B3最后worker仅2行，同址补位3→2→1→3通过；cycles=1仅作功能证据。
本轮没有新七档Native/整模型token结论，20%目标不变且未达成。

仅修改HCA的hc_pre_fused.py与decode_hca.py的原始BF16参数传递；未改Native、共享CSA和工具链。
生产与已测候选除未用导入、DCE消掉的两尾块暂存、注释换行外AST一致，生产显式编译/load通过。
脚本Ruff/shell/diff通过，生产无新增Ruff问题；全仓format.sh ci缺pre-commit，未声称通过。
[源码、逐轮取舍、容量、补位与泳道](hca_pre_ub_20260930/README.md)。
下一轮调度重点是mix与Sinkhorn的同步启动/提前解析联动，先长档，再短档守护；不批量全开同步。
