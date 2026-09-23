# DSV4 Flash CSA：恢复完整 padding 支持的开发计划

状态：方案讨论稿，尚未实施，未跑任何 NPU 任务。
目标读者是继续这项工作的下一位开发者或下一轮会话。

本文只覆盖 padding／dummy 支持本身。用户此前叫停的其他 P3 场景、剩余精度差异排查
不因本计划自动恢复。

## 1. 为什么现在要做

D 侧上线口径是 `cudagraph_mode: FULL_DECODE_ONLY`（见
`dsv4_perf_accuracy_20260827/runtime/decode/run_dp_template.sh`）。在 DP16 下，
PTO CSA 目前有三道 host 闸门会因为 padding 直接退回 Native 或拒绝图模式，
其中最外层的一道是：

```python
# vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/service_config.py:69
if parallel.data_parallel_size > 1 and not should_skip_allreduce_across_dp_group(config):
    raise ValueError("PTO CSA full graphs require Native to skip DP padding; use eager until P4 is validated")
```

结果是 PTO 在真实上线配置下拿不到图模式，性能对照也就没有意义。
恢复 padding 支持是让 PTO 能在上线口径下运行的前置条件。

## 2. Native 的 padding 协议（已逐条核对）

这是本计划的事实基础。PTO 要适配它，**不修改它**。

| 对象 | padding 时的取值 | 出处 |
| --- | --- | --- |
| `slot_mapping[num_tokens:num_tokens_padded]` | `-1` | `model_runner_v1.py:2865` |
| `block_table[num_reqs:num_reqs_padded]` | **`0`**，不是 `-1` | `model_runner_v1.py:2866` |
| `seq_lens[num_reqs:]` | `0` | `model_runner_v1.py:1162` |
| `optimistic_seq_lens_cpu[num_reqs:]` | `0` | `model_runner_v1.py:993` |
| `start_pos_decode[num_reqs_actual:]` | `0` | `dsa_v1.py:1003` |
| DSA 组内 `block_table[num_reqs_actual:num_decodes]` | `0` | `dsa_v1.py:1004` |
| `positions[total_num_scheduled_tokens:]` | **不重置**，保留上一步的陈旧值 | `model_runner_v1.py:1147`、`:1154` 只写 `[:total_num_scheduled_tokens]` |

两条最容易踩的：

- **页表补位填 `0` 而不是 `-1`。** PTO 现有的 6 处 `page >= 0` 保护对页表补位行
  **不成立**，它们只能挡住 slot 为负的写入。0 号页在 vLLM 里是 null block
  （待确认，见第 8 节），读它不越界，但读到的是无关数据。
- **补位 token 的 `positions` 是陈旧值。** 上一步的位置残留在缓冲里，可能是任意大的数。
  PTO 多处用 `position` 推算 compact 行号和页表列号，陈旧位置会让这些索引越界。

另外确认：decode metadata 的 `seq_lens` 是 `self.seq_lens[: self.num_decodes]` 的视图
（`dsa_v1.py:1153`），`num_decodes` 取的是 `num_reqs_padded`。所以
**补位请求在 device 上的 `seq_lens` 恒为 0，且每步由 runner 就地刷新**。
这是第 4 节方案的关键。

### 空 rank 与补位请求不是同一回事

要分开处理，两者的 device 张量特征完全不同：

| | 真实 batch 里的补位请求 | 空 rank 的整批 dummy |
| --- | --- | --- |
| 入口 | `execute_model` 正常路径 | `worker.py:967` → `_dummy_run(6, uniform_decode=True)` |
| `seq_lens` | `0` | `6`（`model_runner_v1.py:3245`） |
| `positions` | 陈旧值 | `127`（`model_runner_v1.py:3275`） |
| `slot_mapping` | 补位段为 `-1` | 整份 `-1`（`model_runner_v1.py:3290`） |
| 输出是否被使用 | 否，但与真实请求同批 | 否，整批丢弃 |

空 rank 的 dummy 看起来就是一个 `seq_len=6`、`position=127` 的合法 B1 请求，
**没法用 `seq_lens == 0` 判别**。好在它整份 slot 都是 `-1`，写入天然安全，
只需保证读不越界、输出有限。日志第 81 节记录过一次 B4 占位输入
（`seq_len=6`、`position=127`、空页表、负 slot）的 warmup/capture 通过，
但那是单层 fixture，不等于 EP16 下的完整路径。

## 3. PTO 现状

### 三道 host 闸门

| 位置 | 现有条件 | 遇到 padding 的行为 |
| --- | --- | --- |
| `service.py:32` `eligible()` | `item.num_actual_tokens != tokens`、`item.num_decodes != batch` | 退回 Native |
| `native_adapter.py:115` | `metadata.num_actual_tokens != tokens` 即报错 | 抛异常 |
| `service_config.py:15` `can_replay_csa_graph()` | 需要 `num_tokens == padded_tokens` | 该步改走 eager |
| `service_config.py:69` | DP>1 且未跳过同步 | 拒绝整个图模式 |

### 已有的 device 保护

6 处缓存／state 写入都有 `page >= 0 && offset >= 0`：
`decode_csa.py:267`、`decode_compressor_ratio4.py:323`、`:395`、
`decode_indexer_compressor.py:275`、`:494`、`:527`。
这些保护挡得住 `slot_mapping = -1`，**挡不住页表补位行的 `0`**。

indexer 已经天然安全：它用 `kv_seq_lens[b] // 4` 算 `visible_count`，
补位请求得到 0，`if visible_count > 0` 直接跳过
（`decode_indexer.py:244`、`:376`、`:479`）。

## 4. 设计决定：设备端有效性从哪来

**本节初稿是只看 PTO 侧推导出来的，已按用户要求改为先对照 Native 的实际做法。**

### Native 自己怎么处理补位请求

Native 的机制分两层，必须分清，因为只有一层在图重放时还起作用：

1. **主机侧，每步在 metadata 构建时执行。** runner 把真实请求数作为 host int
   `num_reqs_actual` 传给 builder（`model_runner_v1.py:2961`），builder 据此
   **就地清零补位请求的 device 张量**（`dsa_v1.py:1002`～`1004`）：

   ```python
   if num_reqs_actual is not None and num_reqs_actual < self.num_decodes:
       self.start_pos_decode[num_reqs_actual:].fill_(0)
       self.block_table[num_reqs_actual : self.num_decodes, ...].fill_(0)
   ```

   这些是对固定地址的就地写，每步都做，发生在重放之前，**图重放照样生效**。

2. **算子侧，逐请求的行为全部来自 device 张量。** decode 路径传给算子的是
   `actual_seq_lengths_query = query_start_loc`、`actual_seq_lengths_key = seq_lens`
   两个 device 张量（`dsa_v1.py:2277`～`2278`），不是 host 标量。

唯一以 host int 形式进入 device 算子的是 `compressor_metadata(..., num_reqs_actual)`
（`dsa_v1.py:1623`）。该调用位于 forward 内，会被一起捕获，**其整数在捕获时就冻结**；
捕获用的 dummy run 没有 padding，所以冻结值等于档位大小。也就是说重放时
Native 的这个算子把补位请求当成真实请求来算，靠的是第 1 层已经把它们的
`start_pos` 和 `block_table` 清零，使算出来的位置落到 0 号页而非真实数据上。

### PTO 应当照着这条线走

结论：**逐请求的判据只能来自 Native 每步就地刷新的 device 张量，不能来自 host 标量。**
这一点 Native 自己也是这么做的，不是 PTO 的特殊限制。

| 方案 | 做法 | 评价 |
| --- | --- | --- |
| A | 只依赖现有 `slot >= 0` 保护 | 不够。挡不住页表补位行的 `0`，也挡不住陈旧 position 导致的越界读 |
| B | 新增有效请求数／掩码入参 | 若做成 host 标量则与 Native 的 `compressor_metadata` 同样会被冻结；若做成 device 张量则需在重放前 H2D，而 PTO forward 在图内无法自行更新 |
| C | 用 `kv_seq_lens[b] == 0` 判定 | 数据来源正确（是 Native 每步刷新的 `self.seq_lens` 视图），但**判据本身是 PTO 自创的**，Native 并不用它来区分补位 |
| D | 用 Native 已经归一化的量：补位请求 `start_pos == 0` 且页表行为 `0` | 与 Native 第 1 层的语义完全一致 |

### 实测定夺：选 C（2026-09-24）

`task_20260924_001111_370250932735` 在真实离线 D 的图模式下采到补位步，
真实 B3 补到 B4，五个 cache group 数据一致：

| 量 | 实测值（索引 3 为补位请求） |
| --- | --- |
| `seq_lens` | `[328, 328, 328, 0]` |
| `start_pos` | `[322, 322, 322, 0]` |
| 页表每行非零列数 | `[3, 3, 3, 0]`，补位行整行为零 |
| `query_start_loc` | `[0, 6, 12, 18, 24]`，补位段按均匀 S 步长补齐 |
| 补位段 `positions` | `[316…321]`，正是**上一步**的位置（真实段为 `322…327`） |

**结论：选 C。** 两者在本样本上都能区分补位，但 D 有歧义而 C 没有：
真实 decode 请求的 `seq_lens` 恒 ≥ S，永远不会是 0；而 `start_pos = seq_lens - S`，
一个刚进入 decode、`seq_lens` 恰为 S 的真实请求，其 `start_pos` 同样是 0，
与补位请求无法区分。因此不采用 D，也不需要在 `native_adapter.py` 里补绑 `start_pos`。

补位段 positions 的实测值同时证实了第 2 节的前提：**补位 token 的 position 是上一步的
陈旧残留**，不是被清零或置为安全值。

### compact 行数：实测 8 行，比初版预测更小

`num_compressed_tokens` 实测为 8（ratio=4 组），而 S0 初版预测 10。原因是
`_num_compressor_metadata_rows`（`dsa_v1.py:606`）用的是**实际** token 数配
**补齐后**的请求数，即 `min(18, 18//4 + 4) = 8`，初版误用了补齐后的 token 数。
探针已按实测更正。行数比误算更小，**越界只会更严重**。

用实测参数（B3→B4、history 328、stale 316）复算四处索引：

| 索引点 | 真实请求 | 补位请求 |
| --- | --- | --- |
| `cmp_slot_mapping` 行 | 0…2 在界内 | **84，张量仅 8 行，越界** |
| `idx_slot_mapping` 行 | 0…2 在界内 | **84，越界** |
| `state_table` 列 | 162…163 在界内 | 156…157 在界内 |
| `ori_block_table` 列 | 6…10 在界内 | 5…10 在界内 |

页表两处此处在界内，是因为本样本的陈旧 position（316）与当前历史（328）同量级，
与 S0 的 uniform 场景一致；混合批次下仍会越界。第 5 节的两类严重度划分成立。

空 rank 的整批 dummy 不走这条判据，单独按第 6 节 S4 处理。

## 5. 风险清单：逐个任务过一遍

按 `decode_csa.py` 的调用顺序。"需要改" 指需要加按请求的有效性跳过。

| 任务 | 补位请求下会发生什么 | 处置 |
| --- | --- | --- |
| `csa_rope_sign` / `build_compact_row_offsets` | 补位请求 `length = 0`，`start = 0 - 6 = -6`，offset 为负；后续消费者若跳过则无影响 | 随消费者 |
| `qkv_proj_rope` | 逐 token 纯算术，读 `freqs` 的补位行（在分配内） | 不改 |
| `csa_cache_writeback` | slot 为 `-1`，已跳过 | 不改 |
| `compressor_ratio4_project` | 逐 token 矩阵乘，无索引风险 | 不改 |
| `compressor_ratio4_pool_projected` | 用陈旧 `position` 算 `logical_pos`，再以 `logical_pos // COMPRESS_STATE_BLOCK_SIZE` 索引 `state_table` 的列 → **可能越列** | **需要改**（`decode_compressor_ratio4.py:209`） |
| `compress_state_commit` | slot 为 `-1`，已跳过 | 不改 |
| `rmsnorm_rope_cache_write` | 用陈旧 `position` 算 `metadata_row`，再读 `cmp_slot_mapping[metadata_row]` → **可能越行** | **需要改**（`decode_compressor_ratio4.py:390`） |
| `indexer_compressor` 写回与 scale commit | 同上的 `metadata_row` 推算 | **需要改**（`decode_indexer_compressor.py:489`、`:522`） |
| `indexer` 全链 | `visible_count = 0` 已跳过 | 不改，但要复核每个入口 |
| `csa_slots_build_valid_qk_plan` | 用陈旧 `position` 算 `v_start`，以 `(v_start + v_lo) // BLOCK_SIZE` 索引 `ori_block_table` 的列 → **可能越列**；且页表补位行为 `0`，`v_page >= 0` 判定为有效 | **需要改**（`decode_sparse_attn_csa.py:216`） |
| `qk_pv` | 受 `valid_block_mask` 控制，上一项修好即可 | 随上项 |
| `decode_o_proj` | 逐 token 动态量化，`INT8_AMAX_EPS` 已防零除 | 不改，但要确认补位行不产生 Inf/NaN |

**一句话结论：真正要改的是三处 compressor／indexer-compressor 的 compact 行推算，
外加 sparse attention 的 SWA 页表计划。其余靠现有保护或本就安全。**

S0 的 predict 已把这四处按严重度分开：compact 行号（B、C）恒越界，属内存安全问题，
优先处理；页表类（A、D）只在陈旧 position 超出当前批次页表容量时越界，
否则读到补位行的 `0` 号页，属正确性问题。详见下一节 S0。

## 6. 分阶段实施

每阶段独立可验证，失败就停在该阶段定位，不跳过。

### S0　证据先行，不改生产代码

**本节初稿写的是"把四处改成显式报错"，那是错的，不可实现**：PTO 的 device 代码
抛不出 Python 异常，越界读只会读到无关数据，不会产生失败现场。改为在 CPU 上
把四处索引公式原样复算，再和张量真实边界比较。常量从生产模块导入，避免与 kernel 漂移。

入口 `dsv4_csa_padding_probe.py`，两种模式：

- `predict`：纯 CPU。按 Native padding 协议构造补位批次，复算四处索引。无需设备与队列。
- `capture`：一张 NPU。用 Native 真实 metadata builder 造出补位后的 device 张量，
  取回 CPU 后用同一套公式复算，并与真实形状比较。

#### predict 结果（2026-09-23，已完成）

真实批 B3 补到 B4、`compact` 行数 10，两种陈旧 position 取值：

| 索引点 | stale = history | stale ≫ history |
| --- | --- | --- |
| `cmp_slot_mapping` 行 | 越界 32775 vs 10 | 越界 |
| `idx_slot_mapping` 行 | 越界 32775 vs 10 | 越界 |
| `state_table` 列 | 在界内 | 越界 65535 vs 133 |
| `ori_block_table` 列 | 在界内 | 越界 4096 vs 11 |

真实请求四处全部在界内，只有补位请求越界。据此把第 5 节的风险分成两类：

- **compact 行号（B、C）恒越界**，与陈旧 position 大小无关。compact metadata 张量按
  **当前步的 token 数**分配（`min(24, 24//4+4) = 10` 行），而行号是
  `offsets + (position+1)//4`，position 一到真实历史量级就跳到三万以上。属内存安全问题。
- **页表类（A、D）只在陈旧 position 超过当前批次的页表容量时越界**。均匀批次下补位
  请求的 position 与真实请求同量级，读到的是补位行里的 `0` 号页——不越界，但是无关
  数据，属正确性问题。混合批次是离线 D 的常态，所以四处都要处理，B、C 优先级更高。

证据：`results/release_csa_padding_20260923/padding_probe_v1/{uniform,mixed}/padding_probe.json`。

注意这里的页表列数按单层 fixture 的分配公式推出；真实 runner 按 `max_model_len`
一次性分配，通常更宽，**A、D 在线上更可能是"读 0 号页"而不是越界**，这一点未实测。

顺带发现：`dsv4_csa_service_dynamic.py` 等既有 fixture 传的是 `num_actual_reqs`，
而 builder 读的是 `num_reqs_actual`（`dsa_v1.py:630`），该 kwarg 一直被静默忽略，
**说明补位路径此前从未被任何测试覆盖**。真实 runner 传的是对的（`model_runner_v1.py:2962`）。

#### capture 待办

predict 的越界结论建立在"compact 张量只有 10 行"上，该行数来自
`_num_compressor_metadata_rows`（`dsa_v1.py:606`）的纯 Python 推算，而张量实际由
device 算子 `compressor_metadata` 产出。需在设备上确认其真实 shape，
并一并回答第 8 节的 Q1、Q5。

### S1　设备端有效性判据

在 `decode_csa.py` 顶层算一次按请求的有效标志（`kv_seq_lens[b] > 0`），
传给需要它的三个子模块。不新增 kernel 入参——`kv_seq_lens` 已经是现有参数。

四处改动：

1. `compressor_ratio4_pool_projected`：`c_idx` 循环体开头跳过无效请求。
2. `compressor_ratio4_cache_write`：`token // S` 得到请求号后跳过。
3. `indexer_compressor` 的写回与 scale commit：同上。
4. `csa_slots_build_valid_qk_plan`：`bias_request` 无效时写
   `valid_block_mask = 0` 并把 `sparse_bias` 填满 `NEG_INF`，让该 token 的
   attention 退化为只剩 sink，输出有限且与真实请求无关。

验收：同一批真实请求，**补位与不补位两种摆法的输出逐 bit 相同**；
整份 allocation（含页 padding 与前后保护区）比对无差异。

### S2　放开 host 闸门

- `native_adapter.py`：`num_actual_tokens` 允许 `<= tokens`；
  按请求的一致性检查改成只对前 `num_reqs_actual` 个请求生效。
- `service.py` `eligible()`：同上放宽，但仍要求真实请求是均匀 S6。
- `service_config.py` `can_replay_csa_graph()`：允许补位后的档位重放 PTO 图。

`validate_configuration` 的 DP 闸门**留到 S5 再动**。

验收：G05——小 BS 放进较大的合法 bucket，eager 与 graph 输出一致。

### S3　单卡 graph 全覆盖

G04（BS 档位切换）、G05（补位）、G06（请求换位／退出／页复用）连续跑。
重点是同一张图在不同补位量下重放，metadata buffer 复用不串数据。

### S4　空 rank dummy

单独验证 `_dummy_run` 整批占位（`seq_len=6`、`position=127`、slot 全 `-1`）：
不写任何 cache／state、输出无非有限值、不越界读。
这一项与 S1 的判据无关，是另一条路径。

若发现越界读，按第 8 节待确认问题里 `metadata_row` 的取值范围单独处理。

### S5　DP 协调

先 DP2 跑通 D01～D05（`(4,40)`、`(40,4)`、`(8,24)`、`(16,32)`、`(0,4)`、`(0,40)` 及连续切换），
再上 DP16。跑之前先把 `should_skip_allreduce_across_dp_group` 的实际返回值、
通信方法和图模式记下来，再确定预期的 padding 量——两种同步模式下预期不同。

这一阶段通过后才拿掉 `service_config.py:69` 的 DP 闸门。

### S6　回到性能对照

S5 通过后，用 FULL_DECODE_ONLY 重跑 Native/PTO 对照，替换日志第 95～97 节
在 eager 下得到的结论。**第 97 节那个 "PyPTO 每次调用重复遍历 AST" 的根因是
eager 特有的，图模式下不成立**，届时要在日志里标注其适用范围，不能直接沿用。

## 7. 验收口径

沿用 `DSV4_FLASH_CSA_VALIDATION_PLAN.md` 已有的编号，不另立标准：

- G05：实际数据少于图容量，检查 padding 和 dummy request（计划第 523 行）
- D01～D05：双卡不均衡负载、空 rank、连续切换（计划第 550～554 行）

通过条件里这几条要逐条给证据，不能只报最终输出一致：

- 补位不误写——整份 allocation 含保护区比对
- 图选择符合实际分支——记录每步实际的 `cudagraph_runtime_mode`
- 两个 rank 的数据不串用
- 无因 replay 触发的重新编译

## 8. 待确认问题

动手前需要先确认，它们会影响方案细节：

1. **0 号页是不是 null block。** 页表补位行填 `0`，若 0 号页是真实可用页，
   补位请求的读会落到别人的数据上（不影响正确性，但会影响"未写区域"比对的口径）；
   若是 null block 则读到的是保留页。本轮没能定位到 vLLM 源码目录，需要在
   `.venv-dsv4-0251rc1` 里确认 `block_pool.py` 的 `null_block`。
2. **空 rank dummy 的 `metadata_row` 会不会越界。** `position=127` 时
   `metadata_row ≈ compact_offsets[0] + 32`，而 dummy 的 compact 行数
   `min(6, 6//4 + 1) = 2`。按 `_num_compressor_metadata_rows`（`dsa_v1.py:606`）
   推算像是会越界，但日志第 81 节记录该用例通过过。需要实测确认，
   不要凭推算直接改 kernel。
3. **补位 token 的 attention 输出会不会带 NaN/Inf 进 MoE。** 跳过 DP 同步时
   不传 `mc2_mask`，补位 token 会真的进入专家路由。S1 的第 4 项把补位 token
   退化成只剩 sink，理论上有限，但要实测。
4. **`num_reqs_actual` 在补位时的实际取值。** `eligible()` 现在检查
   `req.num_reqs_actual not in (None, batch)`，放宽后要重新确定它和
   `num_decodes` 的关系。

## 9. 边界

- 不改 PyPTO、Simpler、PTOAS、PTO-ISA 的计算／运行时实现。
- 不改 Native 的 padding 协议，不新增冗余缓冲绕开它。
- NZ 当前一定不能开：`weight_nz_mode=0`、`enable_kv_nz=false`、
  `VLLM_ASCEND_ENABLE_NZ=0`，Native 侧也一样。上线脚本里的
  `VLLM_ASCEND_ENABLE_NZ=2` 不要照搬。
- 不恢复用户叫停的其他 P3 场景与剩余精度差异排查。
- 所有 NPU 任务走 task-submit 队列。
