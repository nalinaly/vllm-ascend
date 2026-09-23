# DSV4 Flash CSA：后续 session 交接与待办

更新日期：2026-09-23。本文以官方 v0.25.1rc1 迁移后的实际结果为准。
已完成工作提交：`f9bdbb5`，已推送到 `nalinaly/vllm-ascend` 的
`dsv4-flash-pto-v0.25.1rc1` 分支；上一笔生产修正为 `8a4c4e6`。

**当前主线：继续离线 P TP4×DP4/EP16 → D TP1×DP16/EP16 的 CSA 接入与性能对照。**
环境、Native 算子包、H255 离线缓存、Native D16 和 PTO D16 均已跑通，
不要从安装环境、查找 HcPre 或恢复旧仓库重新开始。

**下一项建议先做：在 tests 下补齐排除编译、缓存 IO、预热和观察 hook 的稳态计时，
以已有 H255/B4 bank 建立 Native/PTO 对照。** 然后扩展 P 历史长度与 D batch。
其他 P3/P4 场景、DP padding 改造及剩余数值差异排查仍处于用户要求的暂停状态，
本次交接不代表恢复这些工作。

## 1. 工作目录、分支和环境

| 项目 | 当前值 |
| --- | --- |
| 工作区 | `/data/pyptouser/qinchuanyu/pto-eager` |
| 当前仓库 | 工作区下 `vllm-ascend-dsv4-pto-0251rc1` |
| 分支 / 远端 | `dsv4-flash-pto-v0.25.1rc1` / `https://github.com/nalinaly/vllm-ascend.git` |
| Ascend 基线 | 官方 `v0.25.1rc1`，`9bf964cb4b87c8cd0d6852c41a55b3c29711fa95` |
| vLLM | 官方 `v0.25.1`，`752a3a504485790a2e8491cacbb35c137339ad34` |
| vLLM 源码 | 工作区下 `.cache/migration-v0.25.1rc1/vllm` |
| 激活环境 | 在仓库根目录 `source ../env.sh`，转到 `env-dsv4-0251rc1.sh` |
| Python 环境 | 工作区下 `.venv-dsv4-0251rc1`，Python 3.10 |
| CANN / PTA | CANN 9.0.0；torch `2.10.0+cpu` + torch_npu `2.10.0.post2` |
| PyPTO | 工作区下 `pypto`，debug 分支，`54957491ede07ad5d5015f5e69874f367113cf45` |
| Simpler | 工作区下 `simpler`，debug 分支，`166852bfa658c259478b39e1991a7fd5e7379ac5` |
| 两者必须保持的分支 | `feat/kernel-mode-integration-test` |
| PTOAS / GCC | `.cache/dsv4-toolchain/ptoas-0.61` / 环境脚本选择的 GCC 15.2 |
| 正式模型 | `/data/model/DeepSeek-V4-Flash-0731-w8a8`，75 分片 ModelSlim W8A8 |

torch 的 `+cpu` 后缀是本环境包的实际版本字符串，NPU 由 torch_npu 接入，
已经在 A3 上完成 D16；不要据此替换成 CUDA 包。
正式权重同时含 DSpark draft，Native 加载使用 `mtp.0/1/2` 命名；
不要仅凭 `mtp` 前缀断言缺少 DSpark，也不要恢复成参考 cann_recipe 权重。

旧 `vllm-ascend-dsv4-pto` 目录和 `dsv4-flash-pto` 分支已按用户要求删除。
旧资料保存在当前仓库的 `tests/pypto_test/handoff/legacy_main_18ec20a/` 及历史 results。
Git 公共元数据在工作区 `.git-repositories/vllm-ascend.git`，不要误删。

### 依赖仓库的本地差异必须保留

PyPTO 并非只有上游提交：已有 `torch_npu 2.10.0.post2` 的 shutdown 支持、
对应测试与中英文说明仍在工作区；`_kernel_abi.py` 的 Simpler 版本绑定和
`runtime` 子模块已同步到 `166852bf`。Simpler 自身源码是干净的。
更新前原始差异保存在工作区 `.cache/update-20260923/pypto-before.patch` 和 Git stash。
该备份含旧版本绑定，**不要直接覆盖当前版本**，也不要 reset 掉现有本地差异。

三处运行时 ABI 已核对一致：Python ABI、PyPTO `_torch_npu` 扩展、Simpler
`_task_interface.__build_commit__` 均为 `166852bf`；PyPTO `runtime` 也是该版本。
事实记录见 [dependencies_updated.json](results/release_offline_pd_20260923/dependencies_updated.json)。
后续若用户要求更新依赖，要保留既有差异，并同步检查版本绑定、子模块与已编译扩展；
不要只执行 `git pull` 后沿用旧 `.so`，不要自行修改编译器或运行时实现绕过错误。

### Native 包已经完整恢复

release 原生扩展在仓库 `.cache/csa/native-install/`；包含 HcPre/HcPost 的
14 算子包在 `.cache/csa/csa-native-ops-install/vendors/custom_transformer/`。
源码来自本仓库官方基线的 `csrc/`，没有迁入旧 main 的 Native 算子。
环境脚本已经设置 ATB、custom OPP 和动态库路径。
源码与构建记录见 [native_build](results/release_offline_pd_20260923/native_build/)。
目前没有 HcPre 缺失或算子包仍在编译的阻塞，不需要重编 Native 包。

## 2. 已完成的结果及其边界

下表产物均相对于 `tests/pypto_test/results/release_offline_pd_20260923/`。

| 内容 | 结果与证据 |
| --- | --- |
| HcPre/HcPost、三种 Native metadata | A3 真机通过，任务 `task_20260923_150737_24250853714` |
| P TP4×DP4/EP16，H255 的四种输入 | `smoke_bank/`；16 份缓存、每份 191 个 tensor，覆盖 43 个 target 层及 3 个 DSpark 层；任务 `task_20260923_150829_249108112866` |
| TP4 → TP1 缓存可用性 | 四个 TP 副本的有效前缀直接逐bit一致，`smoke_bank/audit.json`；D 使用 tp0 并按自己的物理页表重映射 |
| Native D TP1×DP16/EP16，B4 | `decode_native_b4_v1/`，64 请求均生成 128 token |
| BF16 norm 入口修正 | `8a4c4e6`：PTO 直接接收 Native BF16 权重，无初始化 FP32 副本；零拷贝检查及完整编译通过 |
| 共享存储报错定位 | `decode_pto_b4_bf16_v3/` 为历史失败；`decode_cache_layout_v1/` 的 336 份层描述符中，672 对重叠全部为同起点、同字节范围 |
| PyPTO PR #2867 更新验证 | 新旧 CPU 复现均归档；更新后接受真实描述符，继续拒绝真正的可写部分重叠；另有 4 项上游 CPU 测试通过 |
| 更新后基础运行时与 CSA 编译 | `updated_eager.xml`：1 项 eager 真机通过；`updated_codegen/report.json`：完整 CSA CPU 编译及 PTOAS 通过 |
| 正式 PTO D16 首次完整运行 | `decode_pto_b4_updated_v4/summary.json`；任务 `task_20260923_175252_286523232409`，exit0，18:04 结束 |

本次 PTO D16 的可确认结论：

- 正式 ModelSlim W8A8、H255、每rank初始 B4，64 个请求共 8192 个输出 token，
  与已有同 rank/case/request 的 Native D16 结果逐 token 完全一致。
- 16rank×21 个 target C4 层全部记录 PTO 调用。
  每rank每层 `pto_tokens24=22`、`pto_tokens18=1`，即 B4/S6 和 B3/S6，
  合计 7728 次 PTO CSA 调用；B3 是本轮实际轨迹，不等于独立完成了 B3 矩阵验收。
- 每rank每层另有 `native_tokens6=8`、`native_tokens1=1`。不要仅根据 token 数
  猜测回退原因；若要分析，结合当步各组 metadata 与 `eligible()` 条件采集证据。
- 首次调用之前的别名拒绝已经消失。修复来自上游
  [PyPTO PR #2867](https://github.com/hw-native-sys/pypto/pull/2867)，无需在 CSA 中复制共享缓存。

这不等于所有层张量逐bit一致，也不等于 P0～P5 全部验收。
本轮 `elapsed_including_io_seconds` 包含冷编译、IO 和观察 hook，**不能算加速比**。
真实 DSpark 已运行，但重复文本 smoke 的接受分布不代表真实业务分布或平均推进 3.8。
执行入口仅设置 `enable_expert_parallel=True`，没有显式开启 EPLB；不能报告 EPLB 已通过。

## 3. 生产实现边界：已经做完的事情不要再重做

| 入口 | 作用 |
| --- | --- |
| `vllm_ascend/models/pypto_deepseek_v4.py` | 继承 release Native 模型，只替换 target 的 21 个 C4 attention；加载后准备每层资源 |
| `vllm_ascend/ops/dsv4_csa.py` | `torch.ops.vllm.dsv4_csa_forward` 派发及 Native 回退 |
| `vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/service.py` | 当前 Native context 的入口判定、两份原生 compact metadata、KV 生命周期 |
| 同目录 `native_adapter.py`、`native_storage.py` | 权重和设备张量绑定、物理页描述符、一次完整 CSA 调用 |
| 同目录 `decode_csa.py` | `decode_csa_tp1_attention_test` 完整生产计算入口；不要因名字含 test 就误认为它只用于测试 |
| 同目录 `service_config.py`、`config.py` | TP1、S6、动态 batch 及容量约束 |

目标是一个完整 PTO CSA 算子，当前已经如此。它内部有多个 device task，
不是要求整个 CSA 只有一个底层 AICore kernel。
此前 8 次外部 PTO 转换调用已消除：token metadata、RoPE 重排、两次 compact 展开、
四次 state 搬运；两份 14 行 state 搬运窗口也已删除。
Native positions、slot、原始页表、紧凑行、cache/state 直接以 device Tensor/视图传入。

Host 仅使用已有 `max_seqlen_q`、请求计数和 Tensor shape/stride 等入口信息，
不读取 device metadata 的值；动态 B/T 由描述符绑定，S 固定为 6。
实现容量 B=1..64，容量上限不代表全部 batch 已验收。
本轮 D16 是 eager，生产 PyPTO runtime 为 `tensormap_and_ringbuffer`，不是 HBG。
不要自行迁移 HBG 或改变全局 DP padding 协议。

release Native 在消费者 stream 生成两份 compact metadata，必须保留对应语义；
它们不是之前已删掉的两次 PTO 展开适配。旧 main 的 ExternalEvent/metadata executor
实现不能直接复制回来。prefill、非均匀 S6、padding/dummy 等不满足入口条件时走 Native。
压缩器的两份 norm 必须直接接收 Native BF16，只在原有 RMS task 内把加载的 tile 转 FP32。

## 4. 可以继续推进的待办，按建议顺序

| 顺序 | 待做事项 | 建议落点与完成条件 |
| --- | --- | --- |
| A1 | H255/B4 稳态性能对照 | 先改 `tests/pypto_test/offline_pd/` 的测试计时；Native/PTO 预热后从相同 bank 初态出发，排除加载、首次编译、首个恢复步骤与观察 hook；记录实际 step、p50/p95、输出 token/s 和峰值显存 |
| A2 | 新基线 profiling 与 PTO 泳道图 | Native/PTO 分开进程采 PyTorch/NPU profile；PTO DFX 独立运行。确认完整 forward、CSA、Indexer、MoE/EP 和 draft 的耗时与依赖，检查 D2H、同步、重复编译和适配调用 |
| A3 | 扩展离线 P 场景 | 继续生成 H4095、32767、131071、131072、131073，每档四种输入；逐档生成、核对有效前缀和层覆盖，再让 D 使用，避免一次盲跑全部长场景 |
| A4 | 扩展 D batch | 每卡 B=1/4/8/16/24/32/40，GBS=16×B；先 B1/B8 确认新路径，再按资源与结果扩展；记录真实 batch、各层 PTO 命中和 Native 回退，不只记名义 BS |
| A5 | 汇总真实 DSpark 与 EP 执行 | 将自然接受长度、实际有效推进、输出数和每rank负载写入结果；对齐 Native/PTO 同场景，再形成可比较的性能报告 |

A1 当前还没有实现专门的稳态计时模式，现有 `decode` 命令只完成一次 `generate`。
可在测试工具中增加预热与独立测量阶段；应确认预热后恢复相同 bank 初态、统计测量窗口内
真正的 decode step 和实际输出 token，不把 6×BS 当作最终生成量。
计划建议 5 轮预热、100 个计时样本，但先确认实际步数是否足够；已有 128 输出 token
通常不足 100 个 decode step。样本不足须如实报告，不凭名义参数宣布采满。

A2 可参考当前 `dsv4_csa_profile.py`、`compare_dsv4_csa_profiles.py` 和工作区
`vllm-ascend-qwen3-14b-pto` 的采集方式，只参考，不照抄模型逻辑。
这些旧单层脚本尚未全部适配 release；`dsv4_csa_profile.py` 仍有旧 fingerprint/hash
流程，不能直接拿旧命令跑本轮。适配时按用户要求去掉新流程中的 hash 检查，
用必要的结果比较和路径/大小清单。新结果保存到新目录，保留旧结果作为历史证据。

A3 的 `full_bank/plan.json` 已有 24 个场景的输入计划，**没有对应长场景缓存**。
更适合按一个新历史长度建一个新 bank 开始；H255 可以直接复用 `smoke_bank`，
不必为了矩阵再生成一遍。

## 5. 尚未完成、但当前必须保留暂停状态的事项

| 项目 | 未完成范围与恢复时注意事项 |
| --- | --- |
| 剩余精度差异 | 旧基线正式权重 B40 step46 在冻结容差内仍有 attention BF16 末位差异；用户已暂停继续追差。旧报告的 35 个 attention BF16 差异不是新 release 已复现的问题 |
| 正式 P0/P2 | 旧基线正式矩阵仅 B4/B40、H131071 通过，旧剩余为16组；迁移后不能简单宣布只剩16组。新 release 的单层 fixture/Native接口、量化ND合同与必要正式矩阵需重新核对 |
| P3 连续轨迹 | 最新正式完整100步验收尚未完成；旧参考100步或旧基线47步不能替代新 release |
| P3 G04～G07 | BS/bucket切换、padding/dummy、请求生命周期/页复用、prefix共享；现有离线 D 关闭 prefix caching，未覆盖这些场景 |
| P3 G08 | 旧 main 跨流事件结果是历史记录；release 的 metadata 生产方式不同，需要按实际机制验证，不能强行引入旧接口 |
| P4 D01～D05 | 只有历史 DP2 metadata 证据，完整 CSA 未验：负载(4,40)/(40,4)、(8,24)/(16,32)、空rank(0,4)/(0,40)、连续切换 |
| DP 同步 padding / full graph | 用户此前明确叫停，尚未决定方案。不能为了让测试通过自行新增冗余缓冲或强行改变 Native padding 协议 |

恢复上述工作需要用户明确重新指派；性能与离线 P/D 工作本身不代表解除暂停。
如果后续接入遇到必须依赖这些行为的新阻塞，先提供具体失败证据、说明影响并讨论方案。

### 最终 P5 的缺口

| 项目 | 本轮覆盖 | 仍缺 |
| --- | --- | --- |
| F01 全模型接入 | H255/B4，全部21个目标C4层，Native/PTO输出token一致 | 其他上下文/BS与必要层级数值验收 |
| F02 实际BS/GBS与graph | GBS64、均衡D16 eager；出现B3收尾 | 其他负载、图容量、实际DP协调及graph验证 |
| F03 PD cache与通知 | 真实P生成离线bank，D恢复与重映射已走通 | 在线传输、网络故障恢复等未验；用户当前选择离线方式，这些不是本轮前置条件 |
| F04 EP/EPLB | EP16真实运行 | 显式开启EPLB、运行中重平衡及与图/生命周期联动未验；不能把enable_expert_parallel等同于EPLB |
| F05 真实DSpark | 正式draft参与生成，有自然接受统计日志 | 多场景结构化接受统计及输出验收；不强制注入平均3.8 |
| F06 稳定性与性能 | 一轮短场景成功退出 | 稳态延迟/吞吐、显存、长时间稳定性及必要异常/超时统计 |

不要照抄旧计划表中的“P5未启动”，也不要把当前短场景成功扩展为“P5全部通过”。

## 6. 可直接使用的路径和命令

以下命令从当前仓库根目录执行。仅在需要相应验证时运行，不因交接重复已有 PASS。
所有 NPU 命令必须走 `task-submit`，CPU 操作无需占卡。

```bash
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
source ../env.sh
task-submit --list
```

已有可直接供 D 使用的 bank：

```text
tests/pypto_test/results/release_offline_pd_20260923/smoke_bank/
  plan.json
  audit.json
  h255_v0/tp0/cache.safetensors
  h255_v0/tp1/cache.safetensors
  ... 共4个输入×4个TP副本
```

本地文件并未提交到 Git；换机器时只 clone 仓库不会带上 bank。
当前 schema1 bank 保存原始快照，D 的 CPU 恢复副本会清除超出有效前缀的尾行；
不要为消除 draft 未来预测行的原始差异而覆盖快照或放宽有效前缀检查。

新 H4095 bank 的示例（路径必须尚未使用，P 完成后再 audit，audit通过后才运行 D）：

```bash
python tests/pypto_test/offline_pd/run.py plan \
  --bank tests/pypto_test/results/release_offline_pd_20260923/bank_h4095_v1 \
  --histories 4095

task-submit --device auto --device-num 16 --max-time 7200 \
  'cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1 && source ../env.sh && python tests/pypto_test/offline_pd/run.py prefill --bank tests/pypto_test/results/release_offline_pd_20260923/bank_h4095_v1 --output tests/pypto_test/results/release_offline_pd_20260923/p_h4095_v1'

python tests/pypto_test/offline_pd/run.py audit \
  --bank tests/pypto_test/results/release_offline_pd_20260923/bank_h4095_v1
```

已有 H255 bank 的 PTO 接入运行示例（仍不是稳态性能命令）：

```bash
task-submit --device auto --device-num 16 --max-time 7200 \
  'cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1 && source ../env.sh && python tests/pypto_test/offline_pd/run.py decode --bank tests/pypto_test/results/release_offline_pd_20260923/smoke_bank --output tests/pypto_test/results/release_offline_pd_20260923/decode_pto_next_v1 --backend pto --batch 4 --decode-tokens 128'
```

Native 用 `--backend native` 和另一个全新输出目录。两个后端各需16卡，按顺序执行。
脚本默认控制地址 `192.168.0.106`、网卡 `enp23s0f3`、DP端口 `29683`；
换机器或端口占用时使用已有 `--host/--nic/--port` 参数，不能挤占他人进程。

CPU 共享缓存回归入口，适合未来依赖更新后按需复验：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python tests/pypto_test/dsv4_csa_shared_storage_repro.py \
  --expect-accepted \
  --layout-dir tests/pypto_test/results/release_offline_pd_20260923/decode_cache_layout_v1 \
  --output /tmp/dsv4_shared_storage_next.json
```

查询排队使用 `task-submit --status <任务ID>`。
不要用短超时的 `task-submit --timeout ... --wait` 当成单纯轮询：此前发现超时会取消
尚未运行的任务。没有必要重复提交同一个等待16卡的任务。
交接时本 session 没有仍在运行或排队的测试；设备空闲情况以新查询为准。

## 7. 用户约束与提交规则

- 后续模型测试全部使用正式 W8A8，不再使用48分片 cann_recipe 参考权重，也不切回纯 BF16。
- CSA 实现放在当前 vLLM-Ascend 仓库；参考对象是 pypto-lib 的
  `models/deepseek_v4_flash_dspark/decode_csa.py::decode_csa_tp1`，不要改 pypto-lib。
- 可以参考 `vllm-ascend-main`、`vllm-ascend-qwen3-14b-pto` 已有接入方式，不能照抄成另一套模型实现。
- 不自行改 PyPTO、Simpler、PTOAS、PTO-ISA 的计算/运行时实现；已有本地环境差异须如实保留。
- 不将 Native BF16 norm 在初始化时转换成 FP32 副本，不恢复已删除的八次外部适配。
- 不过度测试：失败先定位，重跑受影响项；没有实现变化或新疑点时不要重复矩阵。
- 不执行新的 hash/摘要校验。历史报告中的摘要仅保留，后续记录路径/大小并做必要数值比较。
- 所有 NPU 测试通过任务队列，卡够用时可充分利用，但不能绕过队列或停止其他人的任务。
- 沟通、commit说明及新增说明性comment使用中文。用户已要求不做提交检查，不自动运行格式化、
  全量测试或提交钩子；提交带项目要求的 Signed-off-by。
- tests 之外原则上只有足够精简的 PTO 算子与适配代码。测试、诊断、过程记录都放 tests 下。
- 大权重、`.pt/.safetensors`、`.bin/.so/.o`、安装包和重复编译产物留本地，不提交。

本轮 `f9bdbb5` 全部105个文件位于 `tests/pypto_test`，约3.57MB；生产代码没有新增变化。
根目录 `build_output/` 是16个进程的JIT产物，约114MB，未提交；不要 `git add .`。
路径与大小见 [LOCAL_ARTIFACTS.json](results/release_offline_pd_20260923/LOCAL_ARTIFACTS.json)。

## 8. 后续 session 的阅读顺序与记录出口

1. 先读本文、[离线P/D方案](DSV4_FLASH_CSA_OFFLINE_PD.md)及
   [本轮D16汇总](results/release_offline_pd_20260923/decode_pto_b4_updated_v4/summary.json)。
2. 读[迁移说明](BASELINE_MIGRATION_V0251RC1.md)，确认正在使用新分支和 release Native 包。
3. 按需查[验证日志](DSV4_FLASH_CSA_VALIDATION_LOG.md)第89～93节；更早章节用于历史定位，
   不能直接据此宣布新基线 PASS。
4. [原验证计划](DSV4_FLASH_CSA_VALIDATION_PLAN.md)保留完整验收口径，其旧阶段统计不是当前进度表；
   当前待办与暂停范围以本文及用户后续指示为准。

继续将任务ID、具体配置、测试范围、结果、失败原因和下一步记入原验证日志。
每轮使用新结果目录，分别记录接入执行、生成token比较、逐层精度和稳态性能，
避免一个笼统 PASS 覆盖所有含义。当前最有价值的下一步是 A1，而非重做安装或短场景通路验证。
