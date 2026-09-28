> **长期保留的 HCA 性能优化过程日志；本文件为调测信息的唯一权威记录。**
> 与 CSA 侧的 [DSV4_FLASH_CSA_VALIDATION_LOG.md](../../../vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/DSV4_FLASH_CSA_VALIDATION_LOG.md) 对等。
> 接入合同、剩余事项与交接状态以 [DSV4_FLASH_HCA_HANDOFF.md](DSV4_FLASH_HCA_HANDOFF.md) 为准；
> 数据文件与源码快照在 [results/hca_optimization_20260928/](results/hca_optimization_20260928/)。
> 每节记录：修改原因、改了什么、测试配置、实测结果、结论边界、下一步。
> 各节的"当前"、PASS 与权限只对该节的日期、源码与工具链成立，不自动沿用到后续节。
> **尤其注意 CANN 版本分界（见第 9 节）与计时口径（见第 10 节）。**

# DeepSeek-V4 Flash HCA 单层性能优化过程记录

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
128K×B4/B8/B16/B24 + 8K×B24/B32/B40。`submit_hca_seven_tier.sh` 的档位串已改为
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
