# DSV4 Flash CSA：完整执行清单

本文把当前所有待做事项整理成可逐项执行的清单，供后续以目标模式驱动。
每项给出落点、完成判据、依赖和占卡情况；完成判据写成可判真假的形式，
不写"验证一下""确认无误"这类无法判定的措辞。

状态口径：`未开始` / `进行中` / `已完成` / `暂停`（暂停项不得自行恢复）。
截至 2026-09-24，已完成 T1.1～T1.4、T2.2 与 T5.1～T5.4；T3.1 进行中（H4095、H32767 已出）。

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
| T1.5 | graph 覆盖 G04～G06 | 见下方"T1.5 的落点需要改" | 三个用例各自通过；同一张图在不同补位量下重放，metadata buffer 复用不串数据；无 replay 期重新编译 | T1.4 | 16（原定 1，见下） | **待用户定落点** |
| T1.6 | 空 rank 整批 dummy | `offline_pd/run.py` 的 `--rank-decode-tokens`、`observer.py` 的 dummy 计数与 cache 写入实测 | **判据被证伪，待定性**。空转确已发生：rank0 比 rank1 多 7 次 dummy（26 vs 19），与提前 48 token≈8 步吻合。无越界读 ✅；输出无非有限值 ✅（rank1 与基线逐 token 相同）。但**"不写任何 cache／state" 被实测证伪**——`accept_t16_emptyrank_v5` 里 rank0 空转期的 dummy#19/20/21 每次都改写了 `cmp_kv`、`swa`、`compress_state`、`inner_state`（克隆前后均同步，且探测点已越过对照 rank 的 dummy 总数 18）。已排 Native 对照判断这是上游行为还是 PTO 问题 | T1.4 | 16 | **待定性** |
| T1.7 | DP2 跑通 D01～D05 | `tests/pypto_test/dsv4_csa_dp_metadata.py` 扩展到完整 CSA | 六组负载 `(4,40)`、`(40,4)`、`(8,24)`、`(16,32)`、`(0,4)`、`(0,40)` 及连续切换全部通过；两 rank 数据不串用；先记录 `should_skip_allreduce_across_dp_group` 实际返回值、通信方法与图模式，再判定预期 padding 量 | T1.5、T1.6 | 2 | 未开始 |
| T1.8 | DP16 完整验证 | 离线 P/D 入口 | D01～D05 在 DP16／EP16 下通过；DP2 与 DP16 的通信选择分别记录，不互相替代 | T1.7 | 16 | 未开始 |
| T1.9 | 拿掉 DP 图模式闸门 | `service_config.py:69` | T1.8 通过后删除该 `raise`；删除前后各跑一次同配置，确认行为符合预期 | T1.8 | 16 | 未开始 |

用户已指定：**T1.7 的 DP2 必须先跑完再上 T1.8 的 DP16。**

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

### 补位来源的死结（2026-09-24 实测发现）

放开 DP 闸门之后出现一个互相抵消的局面，必须先讲清楚，否则会误以为"补位
已经覆盖到了"：

- DP 闸门要放行，必须 `should_skip_allreduce_across_dp_group == True`；
- 而它为真时 **DP 同步被跳过，各 rank 保留自己的 token 数，不产生 DP 补齐**；
- 档位对齐又消除了档位补齐。

两条来源同时没了。`accept_t14_pto_v5` 实测证实：即便用
`--rank-batches 3 5` 造出 rank 间不均衡，`padded_builds` 仍为 0。

**解法不需要碰"非 6 倍数"那个遗留项**：给一个跳过某些 6 的倍数的档位表即可。
`--capture-sizes 12 30` 下 3 条请求（18 token）只能落到 30 档，补成 5 条、
补 2 条，而每一档仍是整数条请求。这同时构成 T1.3 欠的数值验收——同一负载
分别按 `12 30`（补位）与 `18 30`（不补位）跑，输出必须逐 bit 相同。

**对 G05 的影响**：档位对齐后每个 batch 都落到自己的精确档位，
**单 rank 的档位补齐被彻底消除**（`padding_probe_v2` 实测 `padded_reqs` 全为 0）。
叠加上一节的死结（DP 闸门放行时 DP 补齐也被跳过），G05 只能靠**显式给一个
跳过某些 6 的倍数的档位表**来构造，例如 `--capture-sizes 12 30` 让 18 token
落到 30 档。这不属于"非 6 倍数"那个遗留项——每一档仍是整数条请求。

顺带修好了 MoE 通信选择：`mc2_tokens_capacity` 取自最大档、
`potential_max_tokens` 取 `max(最大档, max_num_seqs*6)`，原先 24 与 30 不等
使 A3 退到 ALLTOALL；对齐后档位为 `[6,12,18,24,30]`，两者相等，MC2 得以选中。

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

### T1.5 的落点需要改（2026-09-24 发现，待用户定）

原定"单卡 graph 覆盖 G04～G06"，但有两条证据表明这个落点走不通：

**一、单卡全链 fixture 已死，且按既定口径不予复活。**
`dsv4_csa_service_dynamic.py` 导入 `dsv4_csa_native_fixture`，后者
`:220` 调用 `builder.enable_device_metadata()`。核实当前 release：
`enable_device_metadata` 与 `take_device_metadata_tasks` **已删除**，
`DeviceMetadataExecutor` 只剩 `service.py:92` 的一句注释。整条链在 import
阶段即失败。用户 2026-09-23 已定：不适配旧接口，取证挂真实生产路径。

**二、G05 单卡已经验不出来。**
档位按 6 对齐后单 rank 的档位补齐被彻底消除（`padding_probe_v2` 实测
`padded_reqs` 全为 0），补位只剩 DP 一个来源。

**建议**：把 T1.5 并入 16 卡的离线 D 生产路径，用已有开关覆盖——
`--stagger` 让活跃 batch 逐档下降覆盖 G04（档位切换）与 G06（请求退出、
换位、合法页复用），DP 补齐覆盖 G05。这样占卡从 1 变成 16，但不需要新建
或复活任何 fixture，且验的是线上真实行为。

是否照此改，请你定。在你定之前 T1.5 不动。

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

### 空 rank 的 dummy 会写 cache（2026-09-24 实测，待定性）

T1.6 原判据写"不写任何 cache／state"，依据是 dummy 的 slot 全为 `-1`、会被
kernel 的 `cache_page >= 0` 守卫挡住。**实测证伪**：`accept_t16_emptyrank_v5`
中 rank0 空转期的 dummy#19/20/21 每次都改写了 `cmp_kv`、`swa`、
`compress_state`、`inner_state` 四个视图。

测量本身已排除两处此前的缺陷：克隆**前后都同步**（v4 只在后面同步，before
快照可能拍在上一步写入未落盘时）；探测点跳过前 18 次 dummy，越过了对照 rank
的 dummy 总数，因此探到的确实是 rank0 自己请求跑完之后的那些。rank1 只有
18 次、未触发探测，构成对照。

已知的相关代码事实：`model_runner_v1.py` 里 `slot_mapping.gpu.fill_(-1)`
**只在非图捕获时执行**；而 CSA 的 compact slot mapping
（`cmp_slot_mapping`／`idx_slot_mapping`）由 Native 的 `compressor_metadata`
算子从 `start_pos` 与 `block_table` 现算，**不在那次 fill 的覆盖范围内**。
这能解释 `cmp_kv` 被写，但 `swa` 走的是应被置 -1 的 `ori_slot_mapping`，
仍无解释——**不要把这条半截推断当成结论**。

**未回答**：写到哪些页、是否落在 0 号 null block、以及该 rank 之后接到新请求
时这些页会不会被复用而出问题。本轮 rank0 空转后没有再接请求，没测到这一层。

已排 Native 同配置对照（`accept_t16_native_control`）：若 Native 也写，
说明是上游 dummy 路径的既有行为，不是 PTO 引入的；若只有 PTO 写，才需要
在 PTO 侧处理。在拿到对照之前不动代码。

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

| ID | 目标 | 完成判据 | 依赖 | 占卡 | 状态 |
| --- | --- | --- | --- | --- | --- |
| T2.1 | FULL_DECODE_ONLY 下重跑 Native/PTO 对照 | 两侧同配置、同 bank 初态；给出每步耗时与设备占用对照；明确标注这是图模式结果 | T1.9 | 16 | 未开始 |
| T2.2 | 标注 eager 期结论的适用范围 | 已完成：验证日志新增第 98 节。第 95～97 节三轮全是 eager（当时图模式起不来），`_resolve_compiled` 按调用次数计费是 eager 特有现象，图模式下只在预热与捕获时走一遍。同时标注了两个未决前提：PTO 的 kernel 下发是否可被图捕获尚在 T1.4 验证中；本机关闭 `fuse_norm_quant` 偏离上线口径 | 无（不依赖 T2.1 数据） | 否 | **已完成** |
| T2.3 | 稳态性能测量（原 A1） | 预热后从相同 bank 初态出发，排除加载、首次编译、首个恢复步骤与观察 hook；记录实际 step 数、p50/p95、输出 token/s、峰值显存；样本不足须如实报告，不按名义参数宣布采满 | T2.1 | 16 | 未开始 |
| T2.4 | 汇总真实 DSpark 与 EP 执行（原 A5） | 自然接受长度、实际有效推进、输出数、各 rank 负载落盘；Native/PTO 同场景对齐 | T2.3 | 16 | 未开始 |
| T2.5 | 决定 PyPTO `_resolve_compiled` 重复遍历 AST 的处置 | 该路径在 PyPTO 内，按约束不自行修改。需用户决定走上游还是本地方案；在此之前只记录，不改 | T2.1 | 否 | **待用户决定** |

`tests/pypto_test/offline_pd/run.py` 已有 `profile`／`profile-export`／`profile-compare`／
`swimlane`／`swimlane-export` 命令，命令行见[离线 P/D 方案](DSV4_FLASH_CSA_OFFLINE_PD.md)。

## 3. T3　场景扩展

| ID | 目标 | 完成判据 | 依赖 | 占卡 | 状态 |
| --- | --- | --- | --- | --- | --- |
| T3.1 | 扩展离线 P 长场景（原 A3） | 逐档生成 H4095／32767／131071／131072／131073，每档四种输入；每档先核对有效前缀与层覆盖再交给 D 使用，不一次盲跑全部。**进度：H4095 ✅（audit PASS，1761 处非致命报告项）、H32767 ✅（audit PASS，报告项为零，3.4G）、H4095 重建对照 ⏳、131071／131072／131073 未开始** | — | 16 | **进行中** |
| T3.2 | 扩展 D batch（原 A4） | 每卡 B=1/4/8/16/24/32/40，GBS=16×B；先 B1/B8 确认新路径再按资源扩展；记录真实 batch、各层 PTO 命中与 Native 回退，不只记名义 BS | T1.9 | 16 | 未开始 |

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

**尚未解释的剩余问题**：H32767 在不开确定性时本来就是 0。既然差异源于归约
顺序，H32767 为何不受影响仍无解释，不要当成已知。

**未测**：开确定性的性能代价没有测量，所以不能据此建议在生产配置里默认开启。

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
