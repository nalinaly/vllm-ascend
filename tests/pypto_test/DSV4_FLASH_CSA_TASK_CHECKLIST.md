# DSV4 Flash CSA：完整执行清单

本文把当前所有待做事项整理成可逐项执行的清单，供后续以目标模式驱动。
每项给出落点、完成判据、依赖和占卡情况；完成判据写成可判真假的形式，
不写"验证一下""确认无误"这类无法判定的措辞。

状态口径：`未开始` / `进行中` / `已完成` / `暂停`（暂停项不得自行恢复）。
截至 2026-09-24，已完成 T1.1、T1.2 与 T5.1～T5.3；T5.4 待用户定。

相关文档：[padding 开发计划](DSV4_FLASH_CSA_PADDING_PLAN.md)、
[跨会话交接](DSV4_FLASH_CSA_NEXT_SESSION_HANDOFF.md)、
[验证计划](DSV4_FLASH_CSA_VALIDATION_PLAN.md)、
[验证日志](DSV4_FLASH_CSA_VALIDATION_LOG.md)、
[离线 P/D 方案](DSV4_FLASH_CSA_OFFLINE_PD.md)。

## 0. 每项开工前都适用的约束

这些是用户的长期要求，不随单项任务改变：

- 不自行修改 PyPTO、Simpler、PTOAS、PTO-ISA 的计算或运行时实现；已有的本地环境差异如实保留。
- 所有 NPU 测试走 `task-submit` 队列；不绕过队列、不停止他人任务；
  不要用短超时的 `--timeout ... --wait` 当轮询，那会取消尚未运行的任务。
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

## 1. T1　padding 支持（当前主线）

方案见 [padding 开发计划](DSV4_FLASH_CSA_PADDING_PLAN.md)。用户已于 2026-09-23 指派恢复。

| ID | 目标 | 落点 | 完成判据 | 依赖 | 占卡 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| T1.1 | CPU 复算四处索引，取得越界证据 | `tests/pypto_test/dsv4_csa_padding_probe.py` | 已完成：真实请求四处全部在界内，补位请求在 compact 行号上恒越界，页表类在陈旧 position 超容量时越界。初版按补齐后 token 数算出 10 行，T1.2 实测为 8 行，公式已更正。证据 `results/release_csa_padding_20260923/padding_probe_v1/{uniform,mixed}/` | — | 否 | **已完成** |
| T1.2 | 设备侧确认 compact metadata 真实形状，并定夺有效性判据 | `offline_pd/observer.py` 的 `offline_begin/end_padding_capture`，`offline_pd/run.py` 的 `padding-capture` 命令 | 已完成，任务 `task_20260924_001111_370250932735`：compact 行数实测 8（初版预测 10，公式已更正）；补位请求 `seq_lens=0`、`start_pos=0`、页表行全零；补位段 positions 实测为上一步残留；据此选定方案 C（`seq_lens == 0`），D 因新请求 `start_pos` 同为 0 而有歧义 | T1.1 | 16 | **已完成** |
| T1.3 | 加入设备端有效性判据并改四处索引 | `decode_csa.py`、`decode_compressor_ratio4.py`、`decode_indexer_compressor.py`、`decode_sparse_attn_csa.py` | 改动前先读 `dsa_v1.py` 中 Native 对同一件事的处理并在提交说明里写明对照结论；同一批真实请求，补位与不补位两种摆法的输出逐 bit 相同；整份 allocation（含页 padding 与前后保护区）无差异 | T1.2 | 1 | 未开始 |
| T1.4 | 放开三道 host 闸门 | `native_adapter.py`、`service.py`、`service_config.py` | G05 通过：小 BS 放进较大合法 bucket，eager 与 graph 输出一致；不再静默回退 Native | T1.3 | 1 | 未开始 |
| T1.5 | 单卡 graph 覆盖 G04～G06 | `tests/pypto_test/` 下新增或扩展 fixture | 三个用例各自通过；同一张图在不同补位量下重放，metadata buffer 复用不串数据；无 replay 期重新编译 | T1.4 | 1 | 未开始 |
| T1.6 | 空 rank 整批 dummy | 同上 | `seq_lens=6`、`position=127`、slot 全 `-1` 的整批占位：不写任何 cache／state、输出无非有限值、无越界读 | T1.4 | 1 | 未开始 |
| T1.7 | DP2 跑通 D01～D05 | `tests/pypto_test/dsv4_csa_dp_metadata.py` 扩展到完整 CSA | 六组负载 `(4,40)`、`(40,4)`、`(8,24)`、`(16,32)`、`(0,4)`、`(0,40)` 及连续切换全部通过；两 rank 数据不串用；先记录 `should_skip_allreduce_across_dp_group` 实际返回值、通信方法与图模式，再判定预期 padding 量 | T1.5、T1.6 | 2 | 未开始 |
| T1.8 | DP16 完整验证 | 离线 P/D 入口 | D01～D05 在 DP16／EP16 下通过；DP2 与 DP16 的通信选择分别记录，不互相替代 | T1.7 | 16 | 未开始 |
| T1.9 | 拿掉 DP 图模式闸门 | `service_config.py:69` | T1.8 通过后删除该 `raise`；删除前后各跑一次同配置，确认行为符合预期 | T1.8 | 16 | 未开始 |

用户已指定：**T1.7 的 DP2 必须先跑完再上 T1.8 的 DP16。**

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

### T1 的待确认问题

动手前需实测，不能凭推算下结论：

| ID | 问题 | 归属 |
| --- | --- | --- |
| ~~T1.Q1~~ | **已答**：vLLM 把 `block_id=0` 保留为 null block（`vllm/v1/core/block_pool.py:188`），初始化时从空闲队列取走并标记 `is_null`，永不分配给任何请求。补位页表行读到的是该保留页，不会串到其他请求的数据 | 已闭环 |
| T1.Q2 | 空 rank dummy 下 compact 行号按推算会越界，但日志第 81 节记录该用例曾通过，两者矛盾 | T1.6 |
| T1.Q3 | 补位 token 的 attention 输出会不会带 NaN/Inf 进 MoE。跳过 DP 同步时不传 `mc2_mask`，补位 token 会真的进入专家路由 | T1.7 |
| T1.Q4 | 放宽闸门后 `num_reqs_actual` 与 `num_decodes` 的实际关系 | T1.4 |
| ~~T1.Q5~~ | **已答**：真实 runner 按 `cdiv(max_model_len, block_size)` 分配页表列（`vllm/v1/worker/gpu_model_runner.py:7039`），比单层 fixture 宽。T1.2 实测样本中 A／D 两处补位请求均在界内，读到的是 0 号页 | 已闭环 |

## 2. T2　性能对照

| ID | 目标 | 完成判据 | 依赖 | 占卡 | 状态 |
| --- | --- | --- | --- | --- | --- |
| T2.1 | FULL_DECODE_ONLY 下重跑 Native/PTO 对照 | 两侧同配置、同 bank 初态；给出每步耗时与设备占用对照；明确标注这是图模式结果 | T1.9 | 16 | 未开始 |
| T2.2 | 标注 eager 期结论的适用范围 | 日志第 95～97 节补上说明：`_resolve_compiled` 每次重复遍历 AST 是 eager 特有现象，图模式下 decode step 捕获一次后只重放，该结论不适用于生产路径 | T2.1 | 否 | 未开始 |
| T2.3 | 稳态性能测量（原 A1） | 预热后从相同 bank 初态出发，排除加载、首次编译、首个恢复步骤与观察 hook；记录实际 step 数、p50/p95、输出 token/s、峰值显存；样本不足须如实报告，不按名义参数宣布采满 | T2.1 | 16 | 未开始 |
| T2.4 | 汇总真实 DSpark 与 EP 执行（原 A5） | 自然接受长度、实际有效推进、输出数、各 rank 负载落盘；Native/PTO 同场景对齐 | T2.3 | 16 | 未开始 |
| T2.5 | 决定 PyPTO `_resolve_compiled` 重复遍历 AST 的处置 | 该路径在 PyPTO 内，按约束不自行修改。需用户决定走上游还是本地方案；在此之前只记录，不改 | T2.1 | 否 | **待用户决定** |

`tests/pypto_test/offline_pd/run.py` 已有 `profile`／`profile-export`／`profile-compare`／
`swimlane`／`swimlane-export` 命令，命令行见[离线 P/D 方案](DSV4_FLASH_CSA_OFFLINE_PD.md)。

## 3. T3　场景扩展

| ID | 目标 | 完成判据 | 依赖 | 占卡 | 状态 |
| --- | --- | --- | --- | --- | --- |
| T3.1 | 扩展离线 P 长场景（原 A3） | 逐档生成 H4095／32767／131071／131072／131073，每档四种输入；每档先核对有效前缀与层覆盖再交给 D 使用，不一次盲跑全部 | — | 16 | 未开始 |
| T3.2 | 扩展 D batch（原 A4） | 每卡 B=1/4/8/16/24/32/40，GBS=16×B；先 B1/B8 确认新路径再按资源扩展；记录真实 batch、各层 PTO 命中与 Native 回退，不只记名义 BS | T1.9 | 16 | 未开始 |

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
| T5.4 | 决定 `dsv4_perf_accuracy_20260827/` 的去留 | 56K、8 个 shell/py 脚本，无凭据，但含内网地址 `172.21.100.73`～`76`（`config.sh`、`docker_run.sh`、`start_decode.sh`、`start_proxy.sh`）。本仓库推送到公开 fork `github.com/nalinaly/vllm-ascend`，是否连同内网拓扑入库由用户决定；可选择脱敏后入库或永久保持本地 | **待用户决定** |

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

如果推进过程中遇到必须依赖这些行为的新阻塞：先提供具体失败证据、说明影响，再和用户讨论方案，
不要为了让测试通过自行新增冗余缓冲或改变 Native padding 协议。

## 7. 建议执行顺序

```
T1.2 → T1.3 → T1.4 → T1.5 ┐
                  └ T1.6 ┴→ T1.7(DP2) → T1.8(DP16) → T1.9
                                                       ├→ T2.1 → T2.2
                                                       │         └→ T2.3 → T2.4
                                                       └→ T3.2 → T4.2
T3.1 可与 T1 并行（只用 P 侧，不依赖 padding）
T5.1～T5.4 随时可做，不占卡
```

关键路径是 T1.2 到 T1.9。T2 和 T4 的多数项都压在 T1.9 之后，
因为在 PTO 拿不到图模式之前，性能数字和 graph 相关验收都没有意义。
