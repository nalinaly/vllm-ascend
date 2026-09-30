# CSA 与 HCA 分支合并验证（2026-09-30）

将 `dsv4-flash-hca-pto-v0.25.1rc1` 合入 `dsv4-flash-pto-v0.25.1rc1`，
保留两侧提交历史与已经提交的过程文件。来源按本次工作开始时的远端版本固定：

- CSA：`041e21cc04333a59def461e59c80f92f2806320c`。
- HCA：`ee88d87fa8153aa31ce389c9dc59b63f34e3253d`；生产实现最近一次修改为 `0742f07c`。
- 共同祖先：`65500b91f263712d4712f803025c0129471f008d`。
- 独立合并目录：`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-csa-hca-merge-0251rc1`。

原 CSA/HCA 目录内未提交的实验不纳入合并，也不改动。PyPTO、Simpler、pypto-lib 未修改。

## 合并处理

1. 保留 CSA 最新融合 O 投影及直接 TaskId 依赖。HCA 需要的分组投影组织函数移到
   `deepseek_v4_flash_hca/o_proj_hc_post.py`，底层矩阵乘继续复用公共实现。
2. QKV 冲突保留 CSA 的整块 gather 改进，接入 HCA 已提交的公共 Q_B 流水和 early 标志。
   公共 Q_B 使用 M128，CSA 精度版及性能版的中间缓冲均按 M128 预留。
3. 公共运行时只初始化一次，沿用 CSA 的运行时选择和 ring 容量配置。
4. 增加 `PyptoCSAHCADeepseekV4ForCausalLM` 联合架构：21 个 C4 attention 半层走 CSA PTO，
   20 个 C128 attention 半层走 HCA PTO；SWA 和 FFN 保持 Native。
   原 CSA、HCA 单独入口保留。
5. 离线入口增加 `--pto-attention both`；服务图闸门、捕获计数和性能解析同时覆盖 41 层。
   报告分别统计 CSA/HCA 区间，严格检查 runtime/worker 数量及 attention/FFN 次序。

普通模型加载可通过 HF override 选择联合架构：

```json
{"architectures": ["PyptoCSAHCADeepseekV4ForCausalLM"]}
```

性能版使用 `PTO_CSA_VARIANT=performance`、`VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0`。
配置仍要求现有 TP1/S6 条件，未支持的 decode 形状回退 Native。

## HCA 增量更新（2026-09-30 下午）

继续合入 HCA 从 `ee88d87f` 到 `b1569fa1` 的六个提交，保留新增的过程文档、脚本和报告。
本次生产变更只有三个文件：

- `deepseek_v4_flash_hca/q_projection_streamed.py`：接入 Q-B 的 L1 K512 搬运与 L0 K128 双缓冲，
  保留四组 head 交接和动态有效行的 compact Acc。
- 公共 `deepseek_v4_flash_dspark_perf/decode_o_proj.py`：接入 HCA 使用的 O-A 两级流水。
  CSA 的调用显式传入 `PIPELINE_OA=False`，保留既有融合 HC_post、直接 TaskId 和 T=96 收尾调度。
- `deepseek_v4_flash_hca/o_proj_hc_post.py`：保留第一次合并时移入的 HCA 分组组织函数，
  补齐 `PIPELINE_OA` 参数并向公共矩阵乘转发 `True`。不能直接用 HCA 分支的整个公共文件覆盖 CSA。

仅按本次接口冲突做定向验证：HCA 完整 root 编译并加载设备库通过，CSA 性能版完整 root
设备代码编译通过；均在 CPU 上完成，没有设备执行。NZ2、atomic0，复用现有调试工具链，
不修改 PyPTO/Simpler。补位 slot 的三个 CPU 用例通过。小型编译报告见
`hca_update_compile.json`，生成源码、库和编译日志留在本地结果目录，不进入 Git。

上午的联合功能结果只适用于更新前版本。14:49 核对发现机器启动时间为 13:04:36，
Native worker 与 engine 进程已消失，原日志止于 12:16 的静态编译等待；没有 Native rank JSON，
也没有 token 对比结果。不能把队列残留的 running 当作任务仍在执行。

重启后 `pto-task.service` 为 not-found/inactive，task-daemon PID 文件对应的进程不存在；
当前用户 `sudo -n -l` 返回需要密码。已向旧任务发送终止请求，但守护进程未运行，尚未收到确认。
已请求恢复队列，不绕过 task-submit 裸跑设备，也不修改其他会话的队列状态。
恢复后先做本次合并的 CSA/HCA 单卡功能，再重新执行 `run_validation.sh` 的联合整网功能和
输出 token 精度；两项通过后才使用 `run_model.sh` 采性能。输出使用新目录，保留上午的旧版证据。

## HCA 资料补同步（2026-09-30 15:24）

远端 HCA 后续新增 `3e31ee2a`，补入七档 Native SK0 三步 profiling 与最终 PTO 泳道
的汇总脚本和说明。已继续合入；此次九个文件均在 tests 下，生产算子没有变化，
前一节的 CPU 编译结果继续适用，不为资料更新重跑设备编译。

15:24 查询时，两条待运行单卡任务 `task_20260930_145445_54704013086`、
`task_20260930_145445_54612631881` 均已变为 `not_found`，没有结果目录；
队列服务仍未运行。因此它们不再是有效待运行任务，队列恢复后须重新提交，
不能根据旧的 pending 记录声称会自动执行。未查明任务记录消失的原因。

## 队列恢复后的新版验证（2026-09-30 16:42）

16:40 确认任务队列服务恢复并能执行。以 `c6128b72` 的合并实现重新提交单卡功能：

- CSA：`task_20260930_164033_353433732387`，退出 0；128K/B16/S6，NZ2/atomic0/det0。
  自身重复调用、A/B/A 图重放、Native/PTO 保护区共 250 项检查通过。
- HCA：`task_20260930_164033_353350118909`，退出 0；同样使用正式权重、128K/B16。
  服务入口、图/eager 及三组保护区共 65 项检查通过。
- 两套报告均为 MEASURED；Native 浮点输出最大绝对差仍为 0.03125，不能称为 Native 精度通过。
  新的小型摘要见 `update_single_card_summary.json`，原始报告和编译产物留在 results 下。

随后启动联合整网任务 `task_20260930_164245_360156022807`，已实际分配 16 张卡。
结果目录为 `results/csa_hca_merge_20260930/validation_h131072_b16_v2`，来源记录固定
`c6128b72`。继续使用 P TP4×DP4 离线 KV、D TP1×DP/EP16、每 rank B16、192 输出 token。
脚本先验证联合 PTO 功能，再运行 Native 并比较输出 token，不采性能。
此处只记录已启动，不预记整网功能、精度或性能通过。

## 新版联合整网功能与 token 观测（2026-09-30 16:56，静态 kernel 未生效）

任务 `task_20260930_164245_360156022807` 完成，退出 0。限定场景为正式 W8A8 权重、
128K 历史、每 rank B16/S6、D TP1×DP/EP16；没有扩大为七档或全部生命周期场景通过。

- 功能：全部 16 个 rank 恢复各 16 条离线 KV，共 256 条；41 层全部选择 PTO，
  每 rank 实际重放 T=96 联合图 33 次，共生成 49,152 个输出 token。
  见 `functional_h131072_b16_v2.json`。
- 精度：Native 与联合 PTO 逐 token 比较 49,152 项，差异为 0；16 个 rank 的 DSpark
  接受计数也一致。见 `tokens_h131072_b16_v2.json`。
- 历史对照：新版与更新前联合 PTO 的输出 token、DSpark 计数一致，
  见 `pto_version_tokens_h131072_b16.json`；该项不是另一份 Native 精度证明。
- 本轮执行代码为 `c6128b72`；后续仅追加过程资料，源码来源检查通过。

以上观测后提交了性能任务 `task_20260930_165707_12203128665`，后因实际静态 kernel 未生效主动终止，退出 130。
使用 `results/csa_hca_merge_20260930/model_h131072_b16_v3` 新目录；脚本已从原始结果
重新确认功能与 token 通过。每侧采无 profiler 的 10 个 model forward 设备样本，
另一次生成原计划采 3 步 Level0 profiling。本轮已终止，不用于正式性能结论。

## 修正热缓存启动缺少本机规模（2026-09-30 17:04）

性能轮日志明确出现 `LOCAL_WORLD_SIZE is not set ... static kernel feature will be disabled`。
回查 v2 功能/精度轮，两侧也出现同一告警。因此 v2 的功能和 49,152 token 一致结果有效，
但仅适用于静态 kernel 实际关闭的配置，**不能放行用户要求的静态编译性能验收**。

根因：`AscendCompiler._configure_backend` 只在冷编译时补 `LOCAL_WORLD_SIZE`；
`AscendCompiler.load` 命中缓存后直接恢复编译图，不执行该初始化。v2 日志确认加载了
npugraph_ex 编译缓存；torch_npu 的 `_is_multicard_env_valid` 因缺变量返回 False。
早晨冷编译曾按每个外部 DP 实例的配置推算为 1，也没有反映本机完整的 16 个 rank。

仅修正离线测试启动器：明确向全部子进程传入本机 `DP×TP` 数量，当前 P TP4×DP4 和
D TP1×DP16 均为 `LOCAL_WORLD_SIZE=16`。不依赖父 shell 或冷编译的副作用，
不修改生产 CSA/HCA 算子、Native 编译器或任何依赖仓库。
worker 新增 `OFFLINE_STATIC_KERNEL` 日志，记录实际静态模块是否加载及成功安装包数；
本机多卡只有 Gloo 组长安装，其余 rank 经 barrier 共享安装并 reselect，不要求每卡安装数非零。
后续须同时检查环境值、实际编译与组长成功安装证据，再接受静态配置下的结果。

启动器四种 NZ 参数的定向 CPU 用例通过，覆盖父环境错误地设置为 2 时仍向全部 16 个
子进程传入 16。两个独立进程 worker 配置用例也通过；首条命令未给子进程传递
`tests/pypto_test` 的 PYTHONPATH，因找不到 offline_pd 而失败，补齐测试环境后通过。
静态安装观测优先读取实际使用的 `npugraph_ex` 模块别名，不另行导入一个空状态模块。
单卡算子功能不受这项多卡启动修正影响，不重复单卡；
接下来重新执行联合功能和 token 对照，确认静态 kernel 生效后再采性能。

## 首次合并已完成的检查

- 配置选择、批次参数传递、性能层映射、离线批次及 token 比较：51 项 CPU 检查通过。
- CSA 精度版完整链 CPU lowering 通过，记录于 `precision_lower.json`；不替代设备精度结论。
- 正式权重单卡 B16/S6、history=131072：
  - CSA：`task_20260930_102939_37201874185`，执行、图重放及保护区检查通过。
  - HCA：`task_20260930_102939_372165426625`，服务入口、图重放及保护区检查通过。
- 上述单卡结果为 `MEASURED`，不是与 Native 逐 bit 相同。CSA/HCA 输出最大绝对差均为
  0.03125，差异详细保留于 `single_card_summary.json`。最终精度按整网输出 token 一致验收。
- 首次两次提交未完成验证：CSA 报告误把新增压缩权重当作四张同向零拷贝权重，修正报告遍历后重跑；
  HCA 被队列补入物理设备号，与脚本内映射到逻辑 0 不符，重提时显式使用 `--device 0`。
  不将这两次失败记作通过。

## 当前执行顺序：功能、精度、性能

用户在第二轮启动期间明确要求先功能、再精度、最后性能。
已主动终止性能任务 `task_20260930_111431_262623832757`（退出 130）。
终止前 16 个 rank 均完成 21 个 CSA 与 20 个 HCA 层的绑定，DSpark 权重加载完成，
已越过此前缺失 AddRmsNormBias 的融合注册阶段并进入静态 kernel 编译。
尚未完成图捕获和 decode，也没有性能采集，不将主动终止记作算子失败。

新的入口 `run_validation.sh` 按以下顺序执行，任何一步失败即停止：

1. **功能**：仅运行联合 PTO 的 decode，不挂计时事件或 profiler。
   `functional.py` 固定检查全部 16 个 rank、每 rank 的完整请求与输出 token 数、
   离线 KV 恢复记录、21 个 CSA + 20 个 HCA 层在实际 B×S 档位的捕获、
   以及生成过程中的 FULL 图重放。功能通过写入 `functional.json`。
2. **精度**：功能通过后才加载 Native，复用功能阶段的 PTO 输出逐 token 对照。
   不要求浮点逐 bit 一致；DSpark 接受率差异单列。写入 `token_comparison.json`。
3. **性能**：本轮不采集。两项通过后再调用 `run_model.sh`，必须传入对应验证结果目录；
   性能脚本会从原始报告重新核对功能和 token，失败时不进入计时。

`run_validation.sh` 现在在开始时保存 `source.json`，记录 Git 提交、bank 路径、history 和 batch；
每侧执行完成后复核执行代码未变化。`run_model.sh` 在读取通过结果之前核对同一记录，
没有来源记录、负载不同、执行代码有未提交修改或验证后已变化时，不进入性能阶段。
比较范围为生产 Python 代码和实际验证入口；仅修改说明、历史报告不会要求重新验证。
两种入口均拒绝复用已存在的结果目录，避免把新旧 rank 报告混在一起。
这项检查只记录代码来源与指定负载，不扫描权重或大快照，也不替代工具链和数值验收。
真实临时 Git 仓库的 CPU 回归用例通过，覆盖文档更新放行、代码更新拒绝、脏代码拒绝、
负载不符拒绝及旧来源记录不可覆盖；两份 shell 入口语法检查通过。

功能检查的 6 项 CPU 用例通过，包含缺 rank、缺 HCA 层、未重放、未恢复完整 KV 和
输出 token 数不足的反例；生产算子和部署配置未变。

旧版任务：`task_20260930_113437_389283610684`，进程已消失，见上方重启说明。
本地结果：`results/csa_hca_merge_20260930/validation_h131072_b16_v1`。
这是 128K/B16 联合整网代表场景，不代表七档、请求生命周期或全部 padding 场景通过。

功能阶段已经通过（11:53）：16 个 rank 全部恢复 16 条请求，共 256 条；
每 rank 在实际 T=96 档位重放联合图 33 次，覆盖全部 41 层，生成 49,152 个 token。
逐 rank 检查记录见 `functional_h131072_b16.json`。已复核原始 `pto/rank*.json`，
16 个 rank 的实际运行配置一致，norm/quant 融合、静态 kernel 与 FULL_DECODE_ONLY 均开启。
Native 精度对照未完成，进程已在机器重启前后消失；不记录 token 一致或性能通过。

## 配置及历史任务

复用正式权重 `/data/model/DeepSeek-V4-Flash-0731-w8a8` 与既有 P TP4×DP4 离线 KV：

```text
/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/release_offline_pd_20260923/h131072_bank
```

首轮为 D TP1×DP/EP16、每 rank B16、S6、128K 历史、NZ mode 2、FULL_DECODE_ONLY。
Native 与联合 PTO 使用相同入场、图档位和显存比例。每侧先预热，再采无 profiler 的连续
10 个 `_model_forward` 设备样本；另一次生成采 3 步 Level0 trace，生成长度均为 192 token。
主性能口径不含 metadata、logits、采样、草稿及步间等待，不能称为端到端吞吐。
最终 token 一致为精度门槛，DSpark 接受率差异另外报告。

首轮任务 `task_20260930_103703_397535810063` 已结束，退出码 1。
日志确认正式权重加载完成、21 个 CSA 与 20 个 HCA 层均绑定成功；11:08 在显存预热的
Native norm/quant 融合 pattern 注册阶段失败，尚未进入正式 decode。原因是启动脚本
仅加载基础 `custom_transformer` 包，漏加载既有的 `csa_template_transformer` 补充包，
导致 `aclnnAddRmsNormBias` / `aclnnAddRmsNormBiasGetWorkspaceSize` 找不到。
此时没有整网 token、图捕获或性能结论。

修正 `run_model.sh`：公共环境之后加载
`results/csa_native_template_20260929/env.sh`，两侧均使用同一补充 vendor；私有 OPP 根
仍分别创建，保留 norm/quant 融合和静态编译。不修改算子实现、不关闭融合、不重新构建依赖。
CPU 动态加载与两个 API 符号检查通过，见 `native_dependency_check.json`。

第二轮结果目录为 `results/csa_hca_merge_20260930/model_h131072_b16_v2`，
随后按上述用户要求主动终止并改跑功能验证。

复跑命令（输出目录应使用新路径）：

```bash
task-submit --device auto --device-num 16 --max-time 5400 \
  'bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-csa-hca-merge-0251rc1/tests/pypto_test/csa_hca_merge_20260930/run_validation.sh /path/to/new/validation 131072 16'
```

功能、精度通过后才使用 `run_model.sh /path/to/new/performance 131072 16 /path/to/passed/validation`。

本轮原始报告、日志、trace 与编译产物本地保存于：

```text
/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-csa-hca-merge-0251rc1/tests/pypto_test/results/csa_hca_merge_20260930/
```

仅提交脚本、说明和小型检查摘要，不提交权重、快照、二进制或大 trace。
