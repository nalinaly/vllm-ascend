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

## 整网对照

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

重提任务 `task_20260930_111431_262623832757`，结果目录为
`results/csa_hca_merge_20260930/model_h131072_b16_v2`；提交时等待 8 卡 CI 释放设备。

复跑命令（输出目录应使用新路径）：

```bash
task-submit --device auto --device-num 16 --max-time 5400 \
  'bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-csa-hca-merge-0251rc1/tests/pypto_test/csa_hca_merge_20260930/run_model.sh /path/to/new/output 131072 16'
```

本轮原始报告、日志、trace 与编译产物本地保存于：

```text
/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-csa-hca-merge-0251rc1/tests/pypto_test/results/csa_hca_merge_20260930/
```

仅提交脚本、说明和小型检查摘要，不提交权重、快照、二进制或大 trace。
