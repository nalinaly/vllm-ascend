# DSV4 Flash CSA：完整执行清单

本文把当前所有待做事项整理成可逐项执行的清单，供后续以目标模式驱动。
每项给出落点、完成判据、依赖和占卡情况；完成判据写成可判真假的形式，
不写"验证一下""确认无误"这类无法判定的措辞。

状态口径：`未开始` / `进行中` / `已完成` / `暂停`（暂停项不得自行恢复）。
截至 2026-09-24，已完成 T1.1～T1.9、T2.1、T2.2、T3.1 与 T5.1～T5.4（共 18 项）；T1.10 低优先级、T2.5 待用户拍板。**T1 的 padding 主线至此全部走通。**

相关文档：[padding 开发计划](DSV4_FLASH_CSA_PADDING_PLAN.md)、
[跨会话交接](DSV4_FLASH_CSA_NEXT_SESSION_HANDOFF.md)、
[验证计划](DSV4_FLASH_CSA_VALIDATION_PLAN.md)、
[验证日志](DSV4_FLASH_CSA_VALIDATION_LOG.md)、
[离线 P/D 方案](DSV4_FLASH_CSA_OFFLINE_PD.md)。

## 0. 每项开工前都适用的约束

这些是用户的长期要求，不随单项任务改变：

- 不自行修改 PyPTO、Simpler、PTOAS、PTO-ISA 的计算或运行时实现；已有的本地环境差异如实保留。
- 所有 NPU 测试走 `task-submit` 队列；不绕过队列、不停止他人任务。
  **轮询一律用 `task-submit --status <id>`，绝不要对排队中的任务用 `--wait`。**
  `--wait` 默认 600 秒超时，超时会把尚未运行的任务直接取消——
  `task_20260924_012911_128588925572` 就是这样被取消的（状态从 pending 变
  not_found，三组验收一个都没跑），白丢一轮排队。等待请用
  `until task-submit --status <id> | grep -qE "completed|failed|cancelled|timeout"; do sleep 30; done`。
- 不执行新的 hash／摘要校验；记录路径与大小，并做必要的数值比较。
- 沟通、commit 说明、新增说明性注释一律用中文；提交带 `Signed-off-by`。
- 不做提交检查、不自动运行格式化、全量测试或提交钩子。
- `tests/` 之外原则上只放足够精简的 PTO 算子与适配代码；测试、诊断、过程记录都放 `tests/` 下。
- 大权重、`.pt`/`.safetensors`、`.bin`/`.so`/`.o`、安装包和重复编译产物留本地不提交；
  不要 `git add .`（`build_output/` 约 114MB 未提交）。
- 模型测试一律使用正式 W8A8，不再使用 48 分片 cann_recipe 参考权重。
- **NZ 当前一定不能开**：`weight_nz_mode=0`、`enable_kv_nz=false`、`VLLM_ASCEND_ENABLE_NZ=0`，
  Native 侧也一样；上线脚本里的 `VLLM_ASCEND_ENABLE_NZ=2` 不要照搬。
- **decode 性能测试一律用 ACL Graph `FULL_DECODE_ONLY`**，不用 eager；
  eager 只用于定位问题，其结论不代表上线表现。
- 不过度测试：失败先定位，只重跑受影响项。

**运行入口**：所有涉及 vllm／torch_npu 的命令都必须先
`source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh`，
它负责激活 `.venv-dsv4-0251rc1` 并设置 CANN 9.0.0、PTOAS、ATB 与
`ASCEND_CUSTOM_OPP_PATH`。系统默认的 `/data/server-toolkits/miniconda3/bin/python`
里没有 vllm，直接用它提交会在 `from vllm import LLM` 处失败
（`task_20260924_004251_10658422228` 即因此白跑一轮）。

## 1. T1　padding 支持（当前主线）

方案见 [padding 开发计划](DSV4_FLASH_CSA_PADDING_PLAN.md)。用户已于 2026-09-23 指派恢复。

| ID | 目标 | 落点 | 完成判据 | 依赖 | 占卡 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| T1.1 | CPU 复算四处索引，取得越界证据 | `tests/pypto_test/dsv4_csa_padding_probe.py` | 已完成：真实请求四处全部在界内，补位请求在 compact 行号上恒越界，页表类在陈旧 position 超容量时越界。初版按补齐后 token 数算出 10 行，T1.2 实测为 8 行，公式已更正。证据 `results/release_csa_padding_20260923/padding_probe_v1/{uniform,mixed}/` | — | 否 | **已完成** |
| T1.2 | 设备侧确认 compact metadata 真实形状，并定夺有效性判据 | `offline_pd/observer.py` 的 `offline_begin/end_padding_capture`，`offline_pd/run.py` 的 `padding-capture` 命令 | 已完成，任务 `task_20260924_001111_370250932735`：compact 行数实测 8（初版预测 10，公式已更正）；补位请求 `seq_lens=0`、`start_pos=0`、页表行全零；补位段 positions 实测为上一步残留；据此选定方案 C（`seq_lens == 0`），D 因新请求 `start_pos` 同为 0 而有歧义 | T1.1 | 16 | **已完成** |
| T1.3 | 加入设备端有效性判据并改四处索引 | 同左四个文件 | **代码已完成**（`4b40896`）：判据取 `kv_seq_lens[b] == 0`，四处均只把已有 `cmp_seq_lens`/`kv_seq_lens` 传入子函数，顶层签名不变（52 参数），无新增入参与缓冲；与 Native 的对照结论写入提交说明；CPU 全链 lowering PASS。**数值验收已通过**（`task_20260924_024451_206700130649`）：同一负载分别按档位 `12 30`与 `18 30` 跑，前者 256 次 build 中 248 次带补位、后者 152 次，补位量相差 96 次，而两轮输出**逐 token 完全相同**。另修复图捕获时 dummy run 的 compact 行越界（见 T1.Q2） | T1.2 | 16 | **已完成** |
| T1.4 | 放开三道 host 闸门 | `native_adapter.py`、`service.py`、`service_config.py`、`platform.py` | **代码完成，验收进行中**（`1815fac`、`3dfb547`）。三道闸门已放开；另发现并修复第四个阻塞——ACL Graph 档位未按 `uniform_decode_query_len` 对齐，导致 MoE 退到 ALLTOALL 使 `should_skip_allreduce_across_dp_group` 为假、触发 DP 闸门。判据：小 BS 放进较大合法 bucket、不再静默回退 Native。**已通过**：PTO 在图模式下完整跑通、输出与 Native 逐 token 相同（`accept_t14_pto_v5`）；补位档位全部进入图重放（`accept_t13_padded`：`replay_padded=62`、`allowed=62`、`rejected=0`），不再静默回退 Native | T1.3 | 16 | **已完成** |
| T1.5 | graph 覆盖 G04～G06 | 由 D01～D05 在 16 卡离线 D 上覆盖，见下方对应关系 | **已完成**：五条判据全部有实测证据。G04／G05／G06 与"同图不同补位量重放不串数据"由 D01～D05 覆盖；最后一条"无 replay 期重新编译"由 `d05_recapture` 实测——采集窗口内统计 `torch.npu.NPUGraph` 新建次数，**16 个 rank 合计为 0**，且该轮用 `--stagger` 让两 rank 跨越不同档位序列（重放 13 vs 20），是最容易触发重新捕获的场景。落点之争已无实际意义：单卡链已死，要验的内容在 16 卡上都验到了 | T1.4 | 16 | **已完成** |
| T1.6 | 空 rank 整批 dummy | `offline_pd/run.py` 的 `--rank-decode-tokens`、`observer.py` 的 dummy／slot／compact 探针 | **已完成**，四条判据全部有实测证据。①空转已证实：rank0 比 rank1 多跑 21 次 dummy（26 vs 5），直接计数；②无越界读：图捕获时的 MTE 越界已修；③输出无非有限值且不影响其余 rank：与 Native 逐 token 相同；④**不写任何真实 cache／state**——主 slot 路径六个 cache group 全为 -1（`accept_t16_slots_v2`），compact 路径经复算（`compact_slot_v2`）只有 2 行、其中 1 行为 -1、另 1 行页号为 **0**，而 0 号页是 vLLM 保留的 null block（`block_pool.py` 初始化即取走并标记 `is_null`，永不分配，见 T1.Q1），两 rank 三次采样一致 | T1.4 | 16 | **已完成** |
| T1.7 | D01～D05 的 DP 验证 | `offline_pd/run.py` 的 `--rank-batches`／`--rank-decode-tokens`／`--stagger` | **已完成**。六组按上线口径（batch 32、`HCCL_BUFFSIZE=1800`、DP 闸门已移除）全部通过，`rejected` 无一例外为 0，且**输出与 Native 逐 token 完全相同，16 个 rank 无一例外**。D01 (4,32)／D02 (32,4) 补位量随负载对称反转；D03a (8,24) 与 D03b (16,32) 补位量完全相同（208/104），说明补的是到全局最大值的差额、与自己提交多少无关；D04 用 `--rank-decode-tokens 16 64` 造真实空转，rank0 `dummy_runs=26` vs rank1 的 5，是直接计数而非耗时推断；D05 用 `--stagger` 让两 rank 跨越不同档位序列（补位 96 vs 80、重放 13 vs 20），同一张捕获图在不同补位量下反复重放无串数据、无重新编译 | T1.9 | 16 | **已完成** |
| T1.8 | DP16 完整验证 | 离线 P/D 入口 | **已完成**：本轮 D01～D05 即在 DP16／EP16 下跑的（驱动硬性 `tp=1, dp=16`），见 T1.7。通信选择记录：档位按 `uniform_decode_query_len` 对齐后 `mc2_tokens_capacity` 与 `potential_max_tokens` 相等，A3 选中 MC2 | T1.7 | 16 | **已完成** |
| T1.9 | 拿掉 DP 图模式闸门 | `service_config.py` | **已完成**（`05ba642`）。顺序按用户 2026-09-24 的决定提前：原计划 T1.8 通过后再删，用户明确"目标肯定是支持 DP 补齐场景的 aclgraph，放开后遇到问题解决具体问题"。移除后 `_sync_metadata_across_dp` 真的 all_reduce，DP 补齐随之产生，D01～D05 才验得到真实场景并全部通过 | — | 16 | **已完成** |
| T1.10 | eager + embedding_tp 的 DP 补齐 | `finegrained_tp_config.embedding_tensor_parallel_size` + eager 用例 | 开启 embedding TP 后在 eager 下跑通：DP 补齐正确产生、PTO 输出与 Native 一致、`_forward_embed_tp` 的静态缓冲容量不被超出。**低优先级**（用户 2026-09-24 定），排在 D01～D05 之后 | T1.9 | 16 | 未开始（低优先级） |

用户已指定：**T1.7 的 DP2 必须先跑完再上 T1.8 的 DP16。**

### embedding_tp 为什么会在 eager 下引出 DP 补齐（T1.10 的背景）

`allow_dp_padding` 的四个条件里有 `or embedding_tp_enable()`，**它与 cudagraph
无关**。原因在建组逻辑（`distributed/parallel_state.py` 的 `_create_or_get_group`）：

```python
rank_grid = torch.arange(world_size).reshape(global_pp_size, global_dp_size, global_tp_size)
group = stage_ranks[chunk * group_size : (chunk + 1) * group_size, tp_idx].tolist()
                    ↑ 切的是 DP 维
```

**embedding TP 组是沿 DP 维切的**，不是沿 TP 维。TP=1／DP=16 且
`embedding_tp_size=4` 时，rank{0,1,2,3} 一组、{4,5,6,7} 一组，组内成员是不同的
DP rank。而 `_forward_embed_tp` 在组内做 **all_gather + reduce_scatter**，
要求组内每个 rank 贡献的 token 数完全一致，否则拼接偏移与切分边界对不上。
所以必须先把这些 DP rank 的 token 数补齐——这就是那个 `or` 的由来。
`oproj_tp_enable` 同理（`_OTP` 用同一个 `_create_or_get_group`）。

连带一条：`_forward_embed_tp` 的静态缓冲按
`capacity = get_potential_max_tokens()` 分配，超了直接 `raise`。
`potential_max_tokens` 正是档位对齐时动过的那个量，**两者是同一个来源**——
若开 embedding TP 而档位配置不当，会直接在这里报错。

当前 `embedding_tensor_parallel_size` 为 0（未开），所以这条路径现在遇不到。

### 生产口径暴露的第一条真实约束：HCCL 缓冲

换成 `--batch 40` 后第一轮直接失败在 MoE 的 MC2 派发算子上，**不是 CSA 的问题**：

```
npu_moe_distribute_dispatch_v2 -> aclnnMoeDistributeDispatchV4，错误码 561002
HCCL_BUFFSIZE_EP is too SMALL, maxBs = 240, h = 4096, epWorldSize = 16,
localMoeExpertNum = 16, k = 6
NEEDED = ((maxBs*8704*16*16) + (maxBs*8192*6)) * 2 = 1043MB, HCCL_BUFFSIZE = 1024MB
```

`maxBs = max_num_seqs * 6`。此前一直用 `--batch 5`（maxBs=30），需求约 130MB，
远在限内，所以从没碰到——**小 batch 把这条真实约束整个绕开了**。

查上线参考（`dsv4_perf_accuracy_20260827/runtime`）：decode 侧
`HCCL_BUFFSIZE=1800`、`--max-num-seqs 32`；prefill 侧 1024。按 32 反推
maxBs=192、需求约 834MB < 1800MB，**上线配置自洽**。驱动已改为 prefill 1024、
decode 1800，与上线一致。

**已定**：用户 2026-09-24 定「以 32 为主验收」，T3.2 的档位表去掉 B=40。

### 验收口径与 DP 补齐的定位（2026-09-24 用户定）

**一、验收一律用生产口径，不用小 batch 图快。**
上线参考脚本是 `--max-num-seqs 32`，用户 2026-09-24 定「以 32 为主验收」；
性能数据另有专门指标（见第 2 节，B=16 / seqlen 8k）。
小 batch 会掩盖问题：`max_num_seqs=5` 时对齐后的档位 `[6,12,18,24,30]` 是稠密的，
每个 batch 精确命中、档位补齐根本不发生——本轮那条"档位对齐消除了补位"的错误
结论就是这么来的。而 `max_num_seqs=32` 对应的档位表是稀疏的，补位照常发生。

**二、DP 该补齐的就补齐，否则性能 GAP 全落在 MoE 的集合通信上。**
这条纠正了"补位是额外开销、能省则省"的直觉：各 rank token 数不齐时 MC2 没法按
统一形状走，代价转嫁到 MoE 的集合通信，反而更贵。**跳过 DP 同步不是优化**，
补齐才是生产该走的路。据此 T1.9 的闸门已提前移除（见上表）。

**三、eager／非 aclgraph 路径也要补用例验证。**
本轮的改动——kernel 的 `seq_lens` 守卫、compact 行号兜底、三道 host 闸门放开、
档位对齐——同样会走到 eager 路径，不能只验图模式。eager 下
`allow_dp_padding` 因 `cudagraph_mode == NONE` 而为假，也不注册捕获档位，
**结构上不产生任何补位**，所以要验的是回归：补位守卫在无补位时是否彻底 no-op、
输出有无变化。已排 `task_20260924_115638_182880813886`：eager 下 PTO、Native
基线、以及 PTO 不均衡负载三组，均用 `--batch 40`。

### 档位为什么必须是 6 的倍数（2026-09-24 定论）

这条解释了 T1.4 遇到的第四个阻塞，也回答了"S 恒为 6 为什么还会有形状问题"。

**padding 不发生在 S 这一维，而在请求数那一维。** 一步 decode 的 token 总数是
`batch × 6`；S=6 来自 `DECODE_SEQ = 1 + DSPARK_SPEC_TOKENS`，从不变动。变的是
batch，而 ACL Graph 要固定形状，所以把 batch 补到最近的档位，补的是**整条假请求**
（实测 `query_start_loc = [0,6,12,18,24]`，第 4 条 `seq_lens=0`）。

**但档位是纯 token 计数。** Native 的布局是 TND——`model_runner_v1.py` 里
"when the layout is TND, the first dimension of hidden_states must equal the last
element of actual_seq_lengths_q"——档位只有 T 一个数字，B 和 S 压扁在一起。
vLLM 默认档位来自通用列表（实测 `[1,2,4,8,16,24]`），与 6 无关；vllm_ascend 里
唯二调整它的地方（950 等距抽样、序列并行按 TP 对齐）也都不按 6 对齐。

档位不是 6 的倍数时，`_pad_query_start_loc_for_fia` 走混合分支，**插入一条
长度为剩余全部 token 的 dummy 请求**：档位 16、2 条真实请求会得到
`query_start_loc = [0, 6, 12, 16]`，最后一条长度 4。此时 `s_dim = 16 // 3 = 5`，
PTO 的所有索引全错——不只是尾部那条。

**处置：修档位，不改 kernel。** pypto-lib 的参考实现把两条路径有意分开——
prefill（`prefill_compressor_ratio4.py`）用 `query_start_loc` 走真 TND，
decode（`decode_compressor_ratio4.py` 等）用 `s_dim = bs // b_dim` 走等长 S，
因为 decode 的 S 恒为 6。把 PTO decode 改成 TND 会与参考实现分叉，收益仅限于
一种上游本可避免的形状。用户 2026-09-24 定：**vllm_ascend 的档位设计不合适，
应贴近 DSpark 的 6 的倍数**。已在 `platform.py` 按序列并行那段的既有写法实现，
并留 `align_decode_capture_sizes` 开关（默认开）以便造反例场景。

### 对齐后补位还在不在：**在**，先前的相反结论已更正

这一节曾写成"档位对齐消除了档位补齐、两条补位来源同时没了"。**那是测试配置
的产物，不成立**，2026-09-24 已更正。

对齐逻辑是把默认档位按 6 向上取整后去重，再补上
`min(max_num_seqs*6, max_num_batched_tokens)`。**得到的档位表是否稠密，
取决于 `max_num_seqs`**：

| `max_num_seqs` | 对齐后档位 | 需补位的 batch |
| --- | --- | --- |
| **5**（本轮测试用） | `[6,12,18,24,30]` | **0/5**，全部精确命中 |
| 8 | `[6,12,18,24,36,42,48]` | 1/8 |
| 16 | `[6,12,18,24,36,42,48,60,66,72,84,90,96]` | 3/16 |
| **40**（生产口径） | `[6,12,18,24,36,42,...,240]` | **9/40** |

生产 batch 下档位表是**稀疏的**（缺 30、54、78…），batch 5→36、9→60、13→84
都要补位，**档位补齐照常发生**。此前 `accept_t14_pto_v5` 测到 `padded_builds=0`，
是因为那轮 `max_num_seqs=5` 恰好落在稠密区间，不能外推。

旁证：`max_num_seqs=40` 算出的档位正是 `[6,12,18,24,36,42,48,...]`，与 probe 里
那份"取自 native_dp_v1 的 DP2 实测档位"完全一致，说明真实 DP2 运行早就是这个形状。

**所以 T1.3 的 kernel 判据与 T1.4 的闸门不是防御性死代码**，生产路径上会真的走到。

仍然成立的一条：DP 闸门放行的前提 `should_skip_allreduce_across_dp_group == True`
会让 DP 同步被跳过，因此**DP 补齐**这一条来源在 T1.9 之前确实不产生；
验收时用 `--capture-sizes` 显式构造档位补齐即可，不必等 T1.9。

顺带修好了 MoE 通信选择：`mc2_tokens_capacity` 取自最大档、
`potential_max_tokens` 取 `max(最大档, max_num_seqs*6)`，原先 24 与 30 不等
使 A3 退到 ALLTOALL，直接触发 PTO 的 DP 闸门；对齐后两者相等，MC2 得以选中。
**这是档位对齐的第二个、与形状无关的理由。**

### T1.2 的取证方式：挂在真实生产路径上

用户 2026-09-23 定：**一切以当前 release 的生产路径为准**，取证走真实离线 D，
不复活旧的单层 fixture。旧 fixture 依赖的 `enable_device_metadata`、
`take_device_metadata_tasks`、`DeviceMetadataExecutor` 在当前 release 中均已删除，
整条链在 import 阶段即失败；按上述口径不予适配，也不作为参考。

挂载点最终选 **`AscendDSAMetadataBuilder.build`**。最初挂 `CSAServiceRuntime.eligible`
是错的，有两个问题：它只在 PTO 后端存在；而且 `can_replay_csa_graph` 一旦发现需要
补位就返回 False，使该步回退 eager 并拿到未补齐的 `BatchDescriptor`，等于把要观察的
padding 自己消掉了。builder 两个后端都会走，不受 CSA 闸门影响。

compact 行数直接读 `decode.num_compressed_tokens`，不额外调用 `compressor_metadata`
算子——那需要与当前 builder 同一层的 impl，取错层会因 `compress_ratio` 不匹配而报错
（`task_20260924_000518_3583841341` 即因此失败），也会扰动本步。

索引复算在 CPU 侧离线做，与 `profile`／`profile-export` 的分工一致。

### 已解决：本机图模式此前无法运行

2026-09-23 两轮 16 卡采集的结论，**影响 T1.2 之后的全部图模式工作**：

| 任务 | 配置 | 结果 |
| --- | --- | --- |
| `task_20260923_233825_305642829457` | `--backend pto --graph-mode eager --batch 4` | exit 0，32 步全部无补位 |
| `task_20260923_234823_326614713564` | `--backend native --graph-mode full_decode_only --batch 5` | exit 1，初始化即失败 |

第一轮证明 **eager 结构上不产生补位**：`allow_dp_padding` 取决于
`cudagraph_mode != CUDAGraphMode.NONE`，eager 下为 False，各 rank 保留自己的
token 数；且 eager 不注册捕获档位。所以补位取证必须在图模式下做。

第二轮暴露图模式本身跑不起来：

```
RuntimeError: Worker failed with error 'aclnnAddRmsNormBias or
aclnnAddRmsNormBiasGetWorkspaceSize not in libopapi.so, or libopapi.so not found.'
```

疑似成因（**未验证，勿当结论**）：`vllm_ascend/utils.py:423` 的
`if not torch.compiler.is_compiling(): bootstrap_custom_op_env()`
在图编译期间跳过 bootstrap，导致 libopapi.so 未加载；eager 下首次调用发生在
编译之外，bootstrap 正常执行。`vllm_ascend/ops/layernorm.py:73` 与 `:100`
在 `enable_custom_op()` 为真时才走 `npu_add_rms_norm_bias`。

交接文档记载本轮 D16 一直是 eager，**本工作区没有任何图模式成功运行的记录**，
与该现象一致。

**已于 2026-09-24 修复。** 实测确认 `aclnnAddRmsNormBias` 在基础 CANN 9.0.0 的
`libopapi.so` 和已构建的 CSA 自定义算子包里都不存在，先前"bootstrap 时序"的猜测被证伪。
真正触发路径是 torch 的 pattern matcher 以 `tracing_mode="real"` 追踪融合 pattern，
等于真的执行一次 `norm_quant_fusion_pass.py:61` 里的 `npu_add_rms_norm_bias`，
于是图编译在建 pattern 阶段就崩；eager 不建 pattern 故从未暴露。

修法是配置开关，不改生产代码：`graph_fusion_pass_manager.py:54` 以
`ascend_compilation_config.get("fuse_norm_quant", True)` 控制该 pass，
测试驱动在图模式下将其置 false。`task_20260924_001111_370250932735` exit 0，
本机首次跑通图模式。

**该项偏离上线口径**：参考脚本所在环境具备该算子、融合为开启状态，本机关闭它
意味着图模式性能不直接等同于线上，T2 的性能对照必须注明这一点。

### T1.5 的判据如何被 D01～D05 覆盖

原计划在单卡上验 G04～G06，但单卡全链 fixture 已死——`dsv4_csa_service_dynamic.py`
导入 `dsv4_csa_native_fixture`，后者调用的 `enable_device_metadata` 与
`take_device_metadata_tasks` 在当前 release 中已删除，整条链 import 即失败，
按既定口径不复活。落点之争最终没有实际意义：要验的内容在 16 卡离线 D 上都验到了。

逐条对应：

| T1.5 判据 | 覆盖它的证据 |
| --- | --- |
| **G04** 六档 BS 之间切换，记录图重选 | D05 用 `--stagger` 让活跃 batch 逐档下降，两 rank 跨越的档位序列不同（`replay_padded` 13 vs 20） |
| **G05** 小 BS 放进较大合法 bucket，检查 padding 与 dummy request | D01～D03：低负载 rank 提交 4～16 条却按 32 条的形状跑，`padded_builds` 208/256 |
| **G06** 请求换位、退出、新增及合法页复用 | D05 的 stagger 使请求在不同步数陆续退出；D04 造出真实空转（rank0 `dummy_runs=26` vs rank1 的 5） |
| 同一张图在不同补位量下重放，metadata buffer 复用不串数据 | D05 两 rank 补位量 96 vs 80、重放次数 13 vs 20，`rejected=0`，且输出与 Native 逐 token 相同 |
| **无 replay 期重新编译** | `d05_recapture`：采集窗口内统计 `torch.npu.NPUGraph` 新建次数，16 个 rank 合计 **0**。窗口开在预热与捕获之后，窗口内不再新建即无重新捕获。测的是行为而非耗时尖峰 |

### T1.8 的执行配方（DP16 六组负载）

驱动已支持按 rank 指定实际提交数（`--rank-batches`，2026-09-24 加入）。
`max_num_seqs` 统一取 `--batch`，所以 `--batch` 要给成各 rank 里的最大值；
不足的 rank 用 `--rank-batches` 逐个指定，其余 rank 按最后一个值补齐。

统一前缀（`B` 为 bank，`R` 为结果根目录）：

```
COMMON="--bank $B --graph-mode full_decode_only --decode-tokens 64 --recompute-scheduler --backend pto"
```

| 用例 | 负载 | 命令追加 |
| --- | --- | --- |
| D01 | `(4,40)` | `--batch 40 --rank-batches 4 40` |
| D02 | `(40,4)` | `--batch 40 --rank-batches 40 4` |
| D03a | `(8,24)` | `--batch 24 --rank-batches 8 24` |
| D03b | `(16,32)` | `--batch 32 --rank-batches 16 32` |
| D04a | `(0,4)` | `--batch 4 --rank-batches 0 4` |
| D04b | `(0,40)` | `--batch 40 --rank-batches 0 40` |
| D05 | 连续切换 | 依次跑上述各组，比对图重选与 metadata buffer 复用 |

D04 的 rank0 提交数为 0：不提交任何请求但仍参与 DP 集合通信，
这既是 D04 的空 rank 路径，也是 T1.6 整批 dummy 的前提。

每组都要先记录 `should_skip_allreduce_across_dp_group` 的实际返回值、
通信方法与图模式，再判定预期 padding 量——清单 T1.7 的判据已有此要求，
DP16 同样适用，不能用 DP2 的结论替代。

### 空 rank 的 dummy 是否写 cache：**探针无效，结论全部撤回**

这一节记录一次失败的测量，保留它是为了避免后人重走。

原本要验 T1.6 的"不写任何 cache／state"。做法是在 dummy 步前后克隆该层的
六个缓存视图并逐元素比较。先后修过两轮缺陷（克隆前补同步、跳过前 18 次 dummy
以避开两个 rank 都有的早期步骤），一度得出结论：四处视图被写，其中
`compress_state` 只有 PTO 写、Native 不写。

**对照实验推翻了整套测量**（`accept_t16_stepcontrol`）。用同一套前后对比逻辑
去量 `execute_model`，钩到的前两次恰好是**空闲步——0 个请求、0 个 token**，
却显示 `cmp_kv` 与 `swa` 各有 114 页、`compress_state` 有 62 页发生变化。
**一个什么都不算的步骤不可能写 114 页**，所以这个前后差异根本不能归因于被测
步骤。

因此：

- 页数、页号、`only_null_block` 判定全部作废；
- **那条"`compress_state` 只有 PTO 写"的二值结论同样作废**——对照组显示
  PTO 的 `compress_state` 在 0-token 步上也会变（62 页／31 页），而没有
  Native 的同类对照数据；
- **T1.6 的"不写任何 cache／state"既未证实也未证伪**。先前写的"已被实测
  证伪"不成立，已删除。

**失效原因已查实（不是推测）。** 读 `decode_cache_layout_v1/rank0.cache_layout.json`
——那是 `offline_cache_layout` 早先采到的真实描述符——CSA 的缓存视图**大面积
共用同一块存储**：

| 重叠对 | 重叠字节 | storage_pointer |
| --- | --- | --- |
| `cmp_kv` ↔ `compress_state` | 676,560,896 | 同为 `20733286793216` |
| `inner_compress_state` ↔ `idx_kv_cache` | 85,891,520 | 同为 `20730199080960` |

`cmp_kv` 与 `compress_state` 是同一块 678MB 分配上的两个视图，偏移分别为
466944 与 233472，各自都覆盖 676MB，几乎完全重叠。

由此三点全部解释通：

1. **任何一处写入都会让多个视图同时"变化"**，因为它们本来就是同一段内存，
   所以"四处视图被写"是假象；
2. **"`compress_state` 只有 PTO 写"是重叠的产物**——写 `cmp_kv` 就会让
   `compress_state` 显示变化；
3. **0-token 步也变**：该分配覆盖所有层的块，而 0 个请求的步骤仍会为 DP 协调
   跑一次内部 dummy 前向，所有层都写，于是全部视图一起显形。

**结论：在这套存储布局下，用"视图是否变化"去归因写入根本不可能成立。**
本节的页数、页号一律不要引用。

### 换成测 slot mapping 之后的结果（有效测量）

改测**输入**而非输出——若 slot mapping 全为 -1，kernel 的 `page >= 0` 守卫
必然挡住，与存储布局无关。中间还绕过一个坑：hook `CSAServiceRuntime.__call__`
在 dummy 步上一次都不触发（实测 `dummy_runs=26` 而 `slot_samples=0`），
因为 6 token 的 dummy 被派发到 12 档做**图重放**，而图重放不跑 Python 前向闸门。
改为在 dummy 之后直接读常驻缓冲——图重放读的就是这些固定地址。

**实测（`accept_t16_slots_v2`）：六个 cache group 的 slot mapping 全部为 -1**
（每组 266 个元素、非负 0 个、max 为 -1）。所以 `model_runner_v1.py` 那句
`slot_mapping.gpu.fill_(-1)` 确实生效，**主 slot 这条写入路径在 dummy 步上
写不进去，已实测确认**。

**仍未覆盖的一条，且捷径已被否掉**：compact slot mapping
（`cmp_slot_mapping`、`idx_slot_mapping`）由 `compressor_metadata` 算子在图内从
`start_pos` 与 `block_table` 现算，不来自被 fill 成 -1 的缓冲。图内产出的张量
Python 侧读不到，于是改读它的**输入** `block_table`，本想论证"全零 → compact
slot 指向 0 号 null block → 无害"。

**实测否掉了这条推理**（`accept_t16_blocktable`）：dummy 步的 `block_table`
**不是全零**，保留着已结束请求的真实页号——六个 group 的非零项分别为
3/15、1/5、7/55、7/55、52/880、28/220，最大页号 115～144。主 slot 仍全为 -1
（那条结论稳），但 compact slot 完全可能算出真实页。

所以这条路径**既没被证明无害、也没被证明有害**。要判真假，只剩两条路：
读图内产出的 compact slot 张量本身，或者比较非重叠的存储区间。

一条未验证的旁证：Native 的 `compressor` 算子同样吃 `state_block_table` 与
`start_pos`，输入一致，行为多半相同——但这是推断，不是测量，不能当结论。

### T1 的待确认问题

动手前需实测，不能凭推算下结论：

| ID | 问题 | 归属 |
| --- | --- | --- |
| ~~T1.Q2~~ | **已答且已修**：`_dummy_run` 在图捕获时把所有 `positions` 填成 127、`seq_lens` 填非零，于是 `(127+1)%4==0` 成立、T1.3 的 `seq_lens>0` 守卫放行，推出的 compact 行号 32 远超该档的 12 行——设备实测报 `MTE instruction DDR address out of range`，PTO 首次图捕获即崩。日志第 81 节"该用例曾通过"的矛盾也因此解开：那次 PTO 根本没执行。已按 compact 表的真实行数（动态维）在三处兜住，`88f59bc` | 已闭环 |
| ~~T1.Q1~~ | **已答**：vLLM 把 `block_id=0` 保留为 null block（`vllm/v1/core/block_pool.py:188`），初始化时从空闲队列取走并标记 `is_null`，永不分配给任何请求。补位页表行读到的是该保留页，不会串到其他请求的数据 | 已闭环 |
| T1.Q3 | 补位 token 的 attention 输出会不会带 NaN/Inf 进 MoE。跳过 DP 同步时不传 `mc2_mask`，补位 token 会真的进入专家路由 | T1.7 |
| T1.Q4 | 放宽闸门后 `num_reqs_actual` 与 `num_decodes` 的实际关系 | T1.4 |
| ~~T1.Q5~~ | **已答**：真实 runner 按 `cdiv(max_model_len, block_size)` 分配页表列（`vllm/v1/worker/gpu_model_runner.py:7039`），比单层 fixture 宽。T1.2 实测样本中 A／D 两处补位请求均在界内，读到的是 0 号页 | 已闭环 |

## 2. T2　性能对照

### T2.1 的结果与归因（2026-09-24）

配置：b=16 / s=6 / TP1 / EP-DP16 / seqlen 8k，3 个稳态 decode step。
b=16 在 KV 并发上限 22.8 之内，所以是干净的满批稳态（96 token/16 请求出现 21 次），
不像 batch 32 那轮被调度器拆成 21+11。

| | Native | PTO | 差异 |
| --- | --- | --- | --- |
| 设备侧总耗时 | 240,013 µs | 347,302 µs | **PTO 慢 45%** |
| kernel 记录数 | 7,520 | 5,378 | PTO 少 28% |

按引擎（µs）：

| 引擎 | Native | PTO | 差异 |
| --- | --- | --- | --- |
| **AI_CPU** | 1,270 | **81,318** | **+80,049** |
| MIX_AIC | 91,468 | 141,966 | +50,498 |
| AI_CORE | 33,407 | 22,929 | −10,477 |
| AI_VECTOR_CORE | 48,768 | 37,601 | −11,166 |
| MIX_AIV | 65,101 | 63,487 | −1,614 |

**PTO 替换掉的 Native 算子确实消失了**：`Compressor_*`（8,104µs）、
`SparseAttnSharedkv_*`（6,685µs）、`VllmQuantLightningIndexer`（3,837µs）
在 PTO 侧均为 0，合计约 18.6ms；矩阵乘类也更快
（`QuantBatchMatmulV3` −10,305µs、`TransposeBatchMatMul` −6,079µs）。

**代价是两个 Native 完全没有的条目**：

| kernel | 次数 | PTO 耗时 |
| --- | --- | --- |
| `simpler_aicpu_kernel_exec_*` | 63 | 79,825 µs |
| `aicore_kernel_mode_0_mix_aic` | 63 | 78,582 µs |

63 = 21 层 × 3 步，**每层每步各一次**。前者跑在 AI_CPU 上，正是 AI_CPU 从 1.3ms
暴增到 81.3ms 的来源——这是 PTO/Simpler 运行时的 kernel 下发路径，不是计算本身。

**两点限定**：窗口含 EP 等待、采集与同步开销，是结构对照而非稳态吞吐结论（那是 T2.3）；
`MoeDistributeDispatchV2` 两侧都有且 PTO 高 6,488µs，但 MoE 不在 PTO 替换范围内，
这部分差异更可能来自各 rank 进入集合通信的时刻不同，不宜直接归因给 PTO。

### 主要性能指标（2026-09-24 用户定）

后续性能数据**以这一组配置为主**：

| 项 | 值 | 说明 |
| --- | --- | --- |
| TP | 1 | 驱动 D 侧硬性 `tp=1`；`service_config.py` 也要求 TP=1 |
| DP | 16 | 16 卡各一个 rank |
| EP | 16 | `enable_expert_parallel=True`，EP world size = TP×DP |
| S | 6 | `DECODE_SEQ = 1 + DSPARK_SPEC_TOKENS`，恒定 |
| **B** | **16** | 每卡 16，GBS = 16×16 = 256 |
| **seqlen** | **8192** | `h8192_bank`，四种输入 |

命令形态：

```
python tests/pypto_test/offline_pd/run.py profile \
  --bank .../h8192_bank --graph-mode full_decode_only \
  --batch 16 --backend {native,pto}
```

档位说明：`max_num_seqs=16` 时对齐后的档位为
`[6,12,18,24,36,42,48,60,66,72,84,90,96]`，16 条请求 = 96 token **精确命中 96 档**，
不产生档位补齐；补位只来自 DP。这与"主要指标"的定位自洽。



| ID | 目标 | 完成判据 | 依赖 | 占卡 | 状态 |
| --- | --- | --- | --- | --- | --- |
| T2.1 | FULL_DECODE_ONLY 下重跑 Native/PTO 对照 | `results/release_csa_perf_8k_20260924/` | **已完成**，按主要指标（b16/s6/TP1/EP-DP16/8k）在图模式下采完整 PyTorch profiling（CPU+NPU、Level1、带 device kernel、`with_stack=False`）。两侧采样窗口完全对齐（第 8/9/10 步，均 96 token / 16 请求），**输出逐 token 相同**。设备侧总耗时 Native 240,013µs vs PTO 347,302µs（**PTO 慢 45%**），kernel 记录数 7,520 vs 5,378。按引擎：AI_CPU 1,270 → **81,318**（主因）、MIX_AIC 91,468 → 141,966；AI_CORE、AI_VECTOR_CORE、MIX_AIV 三项 PTO 均更低。详见下方归因 | T1.9 | 16 | **已完成** |
| T2.2 | 标注 eager 期结论的适用范围 | 已完成：验证日志新增第 98 节。第 95～97 节三轮全是 eager（当时图模式起不来），`_resolve_compiled` 按调用次数计费是 eager 特有现象，图模式下只在预热与捕获时走一遍。同时标注了两个未决前提：PTO 的 kernel 下发是否可被图捕获尚在 T1.4 验证中；本机关闭 `fuse_norm_quant` 偏离上线口径 | 无（不依赖 T2.1 数据） | 否 | **已完成** |
| T2.3 | 稳态性能测量（原 A1） | 预热后从相同 bank 初态出发，排除加载、首次编译、首个恢复步骤与观察 hook；记录实际 step 数、p50/p95、输出 token/s、峰值显存；样本不足须如实报告，不按名义参数宣布采满 | T2.1 | 16 | 未开始 |
| T2.4 | 汇总真实 DSpark 与 EP 执行（原 A5） | 自然接受长度、实际有效推进、输出数、各 rank 负载落盘；Native/PTO 同场景对齐 | T2.3 | 16 | 未开始 |
| T2.5 | 决定 PyPTO `_resolve_compiled` 重复遍历 AST 的处置 | 该路径在 PyPTO 内，按约束不自行修改。需用户决定走上游还是本地方案；在此之前只记录，不改 | T2.1 | 否 | **待用户决定** |

`tests/pypto_test/offline_pd/run.py` 已有 `profile`／`profile-export`／`profile-compare`／
`swimlane`／`swimlane-export` 命令，命令行见[离线 P/D 方案](DSV4_FLASH_CSA_OFFLINE_PD.md)。

### 泳道图必须在 eager 下采

首轮泳道采集（`results/release_csa_perf_8k_20260924/swimlane/`）跟着 decode 的上线口径
用了 `FULL_DECODE_ONLY`，任务队列报 exit=0，但进程内抛了
`RuntimeError: Expected exactly one DFX window, captured 0`——**一个采集窗口都没开成**。

成因是采集窗口挂在 Python 层：`offline_begin_swimlane` 把 `CSAServiceRuntime.__call__`
换成在真实 CSA 调用前后执行 `pypto.torch.begin_dfx()`／`end_dfx()` 的包装。ACL Graph 下
decode 步是图回放，不再执行 Python forward，包装函数一次都进不去。这与本项目里
forward hook 在图回放期间不触发是同一件事。

用户 2026-09-24 明确：**泳道图需要在 eager 模式下采**。这在方法上也成立——芯片泳道记录的是
PTO kernel 内部各流水线（MTE／Vector／Cube／Scalar）的占用，属于 kernel 自身性质，
与它由图回放还是 eager 下发无关，变的只是主机侧下发路径。因此泳道**不跟随** decode
性能测量的 FULL_DECODE_ONLY 口径。`run.py` 已加守卫：`swimlane` 命令要求
`--graph-mode eager`，否则直接报错，与 `padding-capture` 拒绝 eager 相对称。

## 3. T3　场景扩展

| ID | 目标 | 完成判据 | 依赖 | 占卡 | 状态 |
| --- | --- | --- | --- | --- | --- |
| T3.1 | 扩展离线 P 长场景（原 A3） | **已完成**：五档全部生成并通过 audit，每档四种输入。H4095 710M（1761 处非致命报告项）、H32767 3.4G（零）、H131071 13G（零）、H131072 13G（零，2¹⁷ 边界）、H131073 13G（零）。几何／布局／覆盖／history 校验全过，可交给 D 使用。四档对照坐实"只有 H4095 有跨副本差异"，成因见上方确定性一节 | — | 16 | **已完成** |
| T3.2 | 扩展 D batch（原 A4） | 每卡 B=1/4/8/16/24/32，GBS=16×B。**B=40 已按用户 2026-09-24 的决定去掉**——上线口径 `--max-num-seqs 32`（`dsv4_perf_accuracy_20260827/runtime/decode/run_dp_template.sh`），40 超出口径。先 B1/B8 确认新路径再按资源扩展；记录真实 batch、各层 PTO 命中与 Native 回退，不只记名义 BS | T1.9 | 16 | 未开始 |

### T3.1 现状：H4095 bank 已生成，但 audit FAIL（成因已查清，待用户定处置）

`h4095_bank` 四个 case 都已产出，audit 报 `computed-prefix data mismatch`，
错误全在 tp1／tp2／tp3（tp0 是基准）。2026-09-24 用
`dsv4_csa_bank_replica_diff.py` 测出差异分布，结论是**BF16 末位舍入噪声**：

| 张量 | 存储 | 差异元素 | ULP 中位 | p90 | p99 |
| --- | --- | --- | --- | --- | --- |
| `layers.2.swa_cache` | BF16 | 1528/81920 | 1 | 6 | 60 |
| `layers.2.attn` | BF16 | 83/524288 | 1 | 3 | 13 |
| `layers.16.compressor.state_cache` | FP32 | 14336/20480 | 12677 | 106854 | 1057230 |
| `layers.2.indexer.compressor.state_cache` | FP32 | 3584/5120 | 8752 | 57606 | 589815 |

FP32 两项的 ULP 数看着大，但 FP32 的 12677 ULP ≈ 相对 1.5e-3，而 BF16 的
eps = 2⁻⁸ ≈ 3.9e-3——state_cache 是 FP32 存储、BF16 精度计算，中位差异只有
0.4 个 BF16 ULP。`max_abs_diff` 实测 0.024～0.045。差异自层 1 起每层都有，
swa／attn 集中在最后一页的第 24～30 行（最后 7 个位置），state 类则是约 70%
元素各差一点点——都符合"TP/EP 归约顺序不确定"的特征。H255 因序列短未显形。

**更正一条早先的错误判断**：先前记录的"最大相对误差 6.17／28.8，不是纯舍入"
不成立。那个指标用 `|a-b|/max(|a|,|b|)`，在近零值上会饱和到 2.0 附近，
度量本身有问题，不能据此断定非舍入。

**已处置（2026-09-24 用户批准）**：audit 原本要求四个副本逐 bit 相同，但 Native
在 TP+EP 下从不保证这一点，而 D 侧只读 tp0（`connector.py` 里 tp 固定为 0，且
DSA 的 KV 是 MLA 压缩潜变量、跨 TP 复制而非切分，每个副本都是完整一份）。
现在几何／布局／覆盖／history 这些 D 真正依赖的检查保持致命，跨副本数据比较
降为报告项并附 ULP 中位／最大值与 `max_abs_diff`。H4095 audit 已 PASS，
报告 1761 处非致命差异，bank 可交给 D 使用。

**H32767 的结果推翻了"随长度累积"的解释。** 2026-09-24 用当前代码生成
`h32767_bank`（3.4G，四个 case），audit **PASS 且跨副本报告项为零**——四个
TP 副本逐 bit 相同。H32767 比 H4095 长 8 倍，若差异来自舍入噪声随层数／长度
累积，它应该更严重才对，结果反而没有。

**重建结果：差异可复现，不是一次性异常。** 用当前代码重建的
`h4095_rebuild` 仍有 1773 个张量跨副本不同（旧 bank 是 1761）。所以先前
"更可能是那一轮生成时的特定情况"这个猜测被推翻。

现有事实：

| bank | 跨副本差异张量数 |
| --- | --- |
| `h4095_bank`（旧会话产出） | 1761 |
| `h4095_rebuild`（当前代码） | 1773 |
| `h32767_bank` | **0** |

H4095 稳定有差异、H32767 稳定没有，与长度无关，是 H4095 这个配置的特性。
**成因已定性（`task_20260924_025528_23757828067`）：归约顺序不确定。**
用 `--deterministic`（`torch_npu.npu.set_deterministic_level(1)` +
`HCCL_DETERMINISTIC=true`，按用户要求保留 AIV 展开模式）重建的 `h4095_det`，
跨副本差异从 1773 降到 **0**。日志确认 `OFFLINE_DETERMINISTIC level=1` 生效，
且没有出现 libhccl.so 里那条 `Deterministic do not support aiv` 告警。

| 配置 | 跨副本差异张量数 |
| --- | --- |
| H4095 默认 | 1761（旧）／1773（重建） |
| **H4095 + 确定性开关** | **0** |
| H32767 默认 | 0 |
| H131071 默认 | 0 |
| H131072 默认 | 0 |
| H131073 默认 | 0 |

**尚未解释的剩余问题**：**只有 H4095 这一档有差异**，H32767 与 H131071 在不开
确定性时本来就是 0。所以这既不是长度效应也不是普遍现象，而是 H4095 特有的，
成因无解释。已知的只有"开确定性开关能消除它"这一条，不要当成已经理解了。

**性能代价：只有一次粗测，不足以下结论。** 比较 `h4095_rebuild`（不开）与
`h4095_det`（开）两轮 prefill 的 `elapsed_including_io_seconds`：

| | n | 最小 | 中位 | 最大 | 合计 |
| --- | --- | --- | --- | --- | --- |
| 不开 | 4 | 7.05 | 8.28 | 10.39 | 33.05s |
| 开 | 4 | 7.08 | 11.19 | 11.41 | 37.49s |

合计约 +13%。**但每档只有 4 个 case、各跑一次、且含缓存落盘 IO**，最小值几乎
相同（7.05 vs 7.08）而最大值差得多，属噪声很大的单次观测。**不能据此建议是否
在生产配置里默认开启**——要下这个判断需要排除 IO、多次重复的稳态测量。

有价值的一点是两轮**输出完全相同**（都是 `[11799]`），确定性开关不改变结果。

audit 的降级判断不受影响且仍然成立：D 侧只读 tp0，Native 默认口径下不保证
四副本逐 bit 相同，所以不该拿它当门禁。现在多了一条：**确实需要逐 bit 可复现
时，`--deterministic` 是可用手段。**

H255 直接复用 `smoke_bank`，不必为矩阵重新生成。
`full_bank/plan.json` 已有 24 个场景的输入计划但**没有对应长场景缓存**，
更适合按一个新历史长度建一个新 bank 开始。

## 4. T4　P5 最终验收缺口

来自交接文档第 5 节。不要把当前短场景成功扩展为"P5 全部通过"。

| ID | 项目 | 仍缺 | 依赖 | 状态 |
| --- | --- | --- | --- | --- |
| T4.1 | F01 全模型接入 | 其他上下文长度／BS 与必要层级数值验收 | T3.1、T3.2 | 未开始 |
| T4.2 | F02 实际 BS/GBS 与 graph | 其他负载、图容量、实际 DP 协调及 graph 验证 | T1.9 | 未开始 |
| T4.3 | F04 EP/EPLB | 显式开启 EPLB、运行中重平衡及与图／生命周期联动；`enable_expert_parallel` 不等于 EPLB | T1.8 | 未开始 |
| T4.4 | F05 真实 DSpark | 多场景结构化接受统计与输出验收；不强制注入平均 3.8 | T2.4 | 未开始 |
| T4.5 | F06 稳定性与性能 | 稳态延迟／吞吐、显存、长时间稳定性及必要异常／超时统计 | T2.3 | 未开始 |

F03 的在线传输与网络故障恢复不是本轮前置条件——用户当前选择离线方式。

## 5. T5　仓库卫生

| ID | 目标 | 完成判据 | 状态 |
| --- | --- | --- | --- |
| T5.1 | 决定 `offline_pd/run.py` 未提交改动的去留 | 两个开关都保留：`--graph-mode` 默认 `full_decode_only`，把 decode 默认口径从 eager 改成上线口径；`--recompute-scheduler` 默认关闭，保留它是因为在 T1.9 拿掉 DP 闸门之前，它是让 PTO 在 DP16 走图模式的唯一开关 | **已完成** |
| T5.2 | 提交 T1.1 的探针与结果 | `dsv4_csa_padding_probe.py` 与 `results/release_csa_padding_20260923/` 入库 | **已完成** |
| T5.3 | 更正 padding 计划里的 S0 描述 | 原文写"把四处改成显式报错"不可实现——PTO device 代码抛不出 Python 异常，越界读只会读到无关数据。已改为 CPU 复算索引公式，计划正文同步更正并补入 predict 结果 | **已完成** |
| T5.4 | 决定 `dsv4_perf_accuracy_20260827/` 的去留 | **用户 2026-09-24 裁定：不入库，永久保持本地。** 该目录含内网地址 `172.21.100.73`～`76`（`config.sh`、`docker_run.sh`、`start_decode.sh`、`start_proxy.sh`），而本仓库推送到公开 fork `github.com/nalinaly/vllm-ascend`。已加入 `.gitignore` 防止误 `git add`；用户未选择"脱敏后入库"，所以也不要改写地址后再提交 | **已完成** |

## 6. 保持暂停，不得自行恢复

以下项目用户此前明确叫停，**恢复需要用户重新指派**；
性能工作与 padding 工作本身都不代表解除暂停。

| 项目 | 未完成范围 |
| --- | --- |
| 剩余精度差异 | 旧基线正式权重 B40 step46 在冻结容差内仍有 attention BF16 末位差异。旧报告的 35 个 attention BF16 差异不是新 release 已复现的问题 |
| 正式 P0/P2 | 旧基线正式矩阵仅 B4/B40、H131071 通过，旧剩余 16 组；迁移后不能简单宣布只剩 16 组 |
| P3 连续轨迹 | 最新正式完整 100 步验收尚未完成；旧参考 100 步或旧基线 47 步不能替代新 release |
| P3 G07 | Prefix 共享。不在 padding 计划范围内 |
| P3 G08 | release 的 metadata 生产方式与旧 main 不同，需按实际机制验证，不能强行引入旧接口 |
| 非 6 倍数档位的 padding 处理 | 用户 2026-09-24 定：档位就固定在 DSpark+1（即 6）的倍数上，这项作为遗留事项先放着。不要主动去实现让 PTO 吃下 ragged 档位的能力——既不要改 kernel 走 TND，也不要加兼容层。`align_decode_capture_sizes` 开关保留，仅供将来恢复该项时造场景用 |

如果推进过程中遇到必须依赖这些行为的新阻塞：先提供具体失败证据、说明影响，再和用户讨论方案，
不要为了让测试通过自行新增冗余缓冲或改变 Native padding 协议。

## 7. 建议执行顺序

```
T1.2 ✅ → T1.3 ✅代码 → T1.4 ⏳验收中 → T1.5 ⏸待定落点 ┐
                                       └ T1.6 ────────┴→ T1.7(DP2) → T1.8(DP16) → T1.9
                                                                                    ├→ T2.1 → T2.3 → T2.4
                                                                                    └→ T3.2 → T4.2
T2.2 ✅（已完成，不再依赖 T2.1）
T3.1 可与 T1 并行（只用 P 侧，不依赖 padding）：H4095 ✅、H32767 ⏳ 生成中
T5.1～T5.4 ✅ 全部完成
```

关键路径是 T1.4 到 T1.9。T2 和 T4 的多数项都压在 T1.9 之后，
因为在 PTO 拿不到图模式之前，性能数字和 graph 相关验收都没有意义。

**当前唯二需要你拍板的**：T1.5 的落点（见上），以及 T2.5 的 PyPTO
`_resolve_compiled` 处置。其余条目要么在跑、要么依赖关系明确。
