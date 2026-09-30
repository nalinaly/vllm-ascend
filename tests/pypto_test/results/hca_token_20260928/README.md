# HCA 输出 token 精度验证

用户验收口径（2026-09-28）：先精度后性能，最终输出 token 一致即可，不要求中间张量逐 bit 移植。
此前七档的零容差张量 FAIL 是诊断统计，不能直接解释为输出 token 不一致。

当前结果：修正压缩 KV 无效行读取后，8K/B16、TP1×DP/EP8 的 **32768 个输出 token 全部一致**，
DSpark 统计也一致。证据：`h8192_b16_ep8_v3/token_comparison.json` 与 `execution_coverage.json`。
DP/EP16 正式任务 `task_20260928_162843_15654084376` 已完成：**65536 个 token 全部一致**，
DSpark 统计也一致。16 个 rank 均有 20 层 PTO 捕获证据与 44 次真实 `FULL_tokens96` forward，
见 `h8192_b16_ep16_v1/token_comparison.json` 和 `execution_coverage.json`。不外推七档通过。

正式验证场景：8K/B16，P TP4×DP4 的既有离线 KV，D TP1×DP/EP16，DSpark S6。
每个 rank 提交 16 个请求，每请求生成 256 个 token，预期比较 65536 个 token。
同一批请求全部入队并同步后再执行，两侧 temperature=0、ignore_eos=True，模型与缓存初态相同。
只接管 20 个 C128 HCA 层，其余层及 FFN 沿用 Native；不做性能计时和 profiling。

配置：正式 ModelSlim W8A8 权重，NZ mode 2、Native deterministic level 0、PTO atomic0，
FULL_DECODE_ONLY，捕获 S6 档位 6/24/48/96。256 个输出覆盖后续 C128 边界。
单层保护区与图重放检查沿用已有结果；模型结果必须同时提供全部 20 个目标层的 PTO 选择证据。

缓存目录：
`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/release_offline_pd_20260923/h8192_bank`。
模型：`/data/model/DeepSeek-V4-Flash-0731-w8a8`。

DP/EP16 任务 `task_20260928_153629_22495767840` 已在排队阶段撤销：
DP/EP8 预验证先发现功能故障，修复后再提交正式 16 卡验证，避免重复已知失败。
原任务 `task_20260928_145030_464122234` 等待分卡超过 45 分钟，
随后 `task-submit --timeout 45 --wait` 到期自动取消了仍在排队的任务，未执行模型。
已使用原结果目录重提；后续只查询状态，不用带超时的等待命令。

```bash
task-submit --device auto --device-num 16 --max-time 2400 \
  'TASK_DEVICE="$TASK_DEVICE" bash tests/pypto_test/run_hca_decode.sh tests/pypto_test/results/hca_token_20260928/h8192_b16_ep16_v1 8192 16'
```

入口复用 `offline_pd/run.py decode`，新增 `--pto-attention hca`；
比较器使用 `--token-only`，DSpark 接受计数的差异照常记录，但不作为 token 一致的额外门槛。
请求、rank、token 数量缺失以及 token 顺序/内容变化仍判失败。
CPU 比较器 5 项测试通过，包括 token 一致但 DSpark 分布不同，以及 token 改变/缺失的反例。
另在测试 observer 中记录生成窗口的 `_model_forward` 图模式与 token 档位；
FULL 模式至少实际调用一次与全部 20 层 PTO 捕获证据相符的档位，才能认定本轮执行了 HCA PTO。
计数器不读设备数据、不采集时间；CPU 检查覆盖参数透传、排除捕获调用及区分 FULL/NONE。

截至首轮 token 对照结束，生产算子尚未修改。
CPU 只读核对已确认一个候选差异：Native 在 `dsa_v1.forward` 的 BF16 `o_proj_input` 上执行
`inplace_partial_rotary_mul`，HCA 在 FP32 attention 结果上直接 inverse RoPE 后再转 BF16。
CSA 性能版也保留 FP32 直接逆旋转，精度版才复刻中间 BF16 落盘；
因此该差异只记录为 token 分歧时的定位候选，不据此强行改变数值路线或宣称 token 错误。
大权重、KV 快照和二进制继续本地保留，不做 hash 校验。

## 分卡等待期间的 DP/EP8 预验证

全卡任务长期等待，另提交 `task_20260928_154509_302261220275` 使用 8 张空闲卡。
仍为 TP1、8K/B16、每请求 256 token，预期比较 32768 个 token，
只将 DP/EP 改为 8；两侧使用相同配置、正式权重和离线 KV。
这轮只能作为 HCA token 预验证，不能代替排队中的 DP/EP16 结果，也不比较性能。

`offline_pd/run.py` 新增测试参数 `--decode-dp {8,16}`，默认 16，8 仅允许 HCA decode；
rank JSON 明确记录本轮 DP，其他测试命令和 P4×DP4 保持 16 卡。
CPU 参数检查确认默认 DP16 和显式 DP8 分派；实际显存与启动状态以设备结果为准。

首轮 v1 已退出 1：主模型加载和 20 个 HCA PTO 层绑定成功，草稿模型加载及显存预热完成，
随后在分配 KV 时报告可用额度 −0.15 GiB；未进入 decode，没有 token 结果。
改用两侧相同的 `gpu_memory_utilization=0.95` 重提 v2：`task_20260928_155014_3884197105`。
新增测试 CLI 显式传递、记录显存预算；DP/EP16 默认仍为 0.9。

```bash
task-submit --device auto --device-num 8 --max-time 1200 \
  'TASK_DEVICE="$TASK_DEVICE" bash tests/pypto_test/run_hca_decode.sh tests/pypto_test/results/hca_token_20260928/h8192_b16_ep8_v2 8192 16 8'
```

v2 已完成两侧运行，token 比较 **FAIL**：32768 个 token 中 32640 个不同，无结果缺项。
所有请求第一个 token 相同；PTO 从第二个 token 起持续输出 0，Native 正常生成。
每个 rank 均观察到 45 次 `FULL_tokens96`，全部 20 层均有匹配的 `pto_tokens96` 捕获证据。
PTO 侧相同输入的重复请求及对应 rank 输出一致，但这种一致显然不能当作精度通过。
证据：`h8192_b16_ep8_v2/token_comparison.json` 及两侧 8 份 rank JSON。

当前定位候选：HCA 压缩 KV 整页 gather 可能读取有效长度之后的未初始化行；
后续即使概率为零，PV 中 `0 × NaN` 仍可污染结果。
已提交单卡反例 `task_20260928_160139_61946828093`：正式第 3 层、B16/H8192，
只把本步不可见的压缩行填为 NaN，对照 Native/PTO 输出及保护区。
任务提交时排队，尚未据此宣称根因确认。

源码核对已确认原 gather 不使用有效长度限制读入。已在 HCA 内核加入
`seq_lens // 128` 上限：跳过无效整页，部分页用 `gather_row(valid_shape=...)` 只搬有效行，
片内其他位置保持零，valid mask 同步限制。没有新增外部转换、清空 Native cache 或修改依赖仓库。
CPU 整层设备编译 PASS：`masked_gather_build/report.json`。

上述单卡任务及部分页用例 `task_20260928_160505_68626121434` 在代码修正时都仍未开始，
因此它们首次运行即验证修正版，不能作为旧实现的设备复现结果。
部分页用例 H8320 对应 65 个有效压缩行，第三页仅一行有效。

修正后整模型任务 `task_20260928_161034_88223618843` 已提交，结果目录 `h8192_b16_ep8_v3`。
只运行 PTO，Native 复用 v2 的相同权重、bank、EP8/B16、256 token、NZ2/det0 和 95% 显存配置结果；
比较报告记录两侧来源目录，避免重复加载未改动的 Native。

两个单卡用例已退出 0：H8192 毒化 1024 行，H8320 毒化 1008 行，
两侧输出非有限值均为 0，Native/PTO 写保护全部通过。
输出最大绝对差均为 0.03125，仍保留零容差差异统计，不将其当作 token 验收。
证据：`poison_b16_v1/report.json`、`poison_partial_b16_v1/report.json`。
这说明修正版不再被这些无效行中的 NaN 污染；整模型 token 故障是否完全消除，仍等 v3 结果。

v3 已完成并 PASS，任务退出 0：比较 8 个 rank、每 rank 16 个请求、每请求 256 token，
共 32768 个 token，差异 0，DSpark 统计差异 0，结果缺项 0。
8 个 rank 各有 44 次 `FULL_tokens96`，全部 20 层具有对应 PTO 捕获证据，
详细记录见 `h8192_b16_ep8_v3/execution_coverage.json`。
H8192 毒化用例的两侧输出分别与旧有限填充快照逐 bit 相同，
证据 `poison_b16_v1/finite_fixture_comparison.json`；没有为追求中间逐 bit 相同而改动数值路线。

已重新提交 DP/EP16 正式验证 `task_20260928_162843_15654084376`，
本次两侧重新运行，显存预算回到 90%，预计比较 65536 个 token。
后续先读取任务状态与 `h8192_b16_ep16_v1/token_comparison.json`，性能优化继续暂停。
