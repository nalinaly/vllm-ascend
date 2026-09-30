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

## 已完成的检查

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

功能检查的 6 项 CPU 用例通过，包含缺 rank、缺 HCA 层、未重放、未恢复完整 KV 和
输出 token 数不足的反例；生产算子和部署配置未变。

当前任务：`task_20260930_113437_389283610684`。
本地结果：`results/csa_hca_merge_20260930/validation_h131072_b16_v1`。
这是 128K/B16 联合整网代表场景，不代表七档、请求生命周期或全部 padding 场景通过。

功能阶段已经通过（11:53）：16 个 rank 全部恢复 16 条请求，共 256 条；
每 rank 在实际 T=96 档位重放联合图 33 次，覆盖全部 41 层，生成 49,152 个 token。
逐 rank 检查记录见 `functional_h131072_b16.json`。已复核原始 `pto/rank*.json`，
16 个 rank 的实际运行配置一致，norm/quant 融合、静态 kernel 与 FULL_DECODE_ONLY 均开启。
Native 精度对照正在初始化，此时不记录 token 一致或性能通过。

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
