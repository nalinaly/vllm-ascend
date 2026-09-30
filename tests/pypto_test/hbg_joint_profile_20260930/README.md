# 单卡 HBG CSA→HCA 联合采集与 checksum 消融（2026-09-30）

已完成 128K/B4、三个 step 的 PyTorch profiling 与同轮联合泳道。
去除 replay 的重复全包 FNV 扫描后，联合设备 replay 均值从
224361.40 降至 207478.31 μs，减少 16883.09 μs（7.52%）。
**真实算子的主要 GAP 仍在核内任务之外，不能把空图的 checksum 占比直接外推。**

## 下载

同一文件夹：[HBG_CSA_HCA_128K_B4_3Steps_20260930](../results/HBG_CSA_HCA_128K_B4_3Steps_20260930/)。

| 文件 | 内容 |
| --- | --- |
| `01_Before_128K_B4_CSA-HCA_HBG_PyTorch_3Steps.json` | 保留 checksum，完整 PyTorch/CANN trace |
| `02_Before_128K_B4_CSA-HCA_HBG_Swimlane_3Steps.json` | 与 01 同轮的六次 launch、AIC/AIV task 和 AICPU 调度 |
| `03_After_128K_B4_CSA-HCA_HBG_PyTorch_3Steps.json` | 去除逐次 replay checksum，完整 PyTorch/CANN trace |
| `04_After_128K_B4_CSA-HCA_HBG_Swimlane_3Steps.json` | 与 03 同轮的联合泳道 |
| `evidence.json` | 环境、队列任务、原始路径、逐次耗时和状态检查 |

JSON 可用 Perfetto/Chrome trace viewer 打开。PyTorch 文件保持原始格式；
泳道使用 Simpler 官方转换器，按每个 launch 匹配其对应的 CSA/HCA 依赖图，
没有人为平移各段、压缩空白或合成 task 时间。每次 CSA 有 712 条核内记录，
HCA 有 414 条；两轮每个 step 的逐 task、逐 AIC/AIV 类型记录数均与依赖图的
block_num 一致，无丢失 launch boundary。

泳道中的 `scheduler+incore` 标记从 scheduler 初始化处开始，
**不包含此前的 graph packet 验证/恢复，也不包含末尾退出开销**。
不能把该标记当成完整算子耗时。完整 runtime 区间看 PyTorch trace 中按时序
交替出现的六个 `simpler_aicpu_kernel_exec_*`。
CANN 解析器报告了 ACL→NPU flow 关联失败；NPU 区间和三个 step 均存在，
但不承诺 Host/NPU 连线完整，也没有手工制造这类连线。

## 测试口径与结果

- 单卡 TP1、B4、history=131072，每请求出 5 验 6，共 24 个 query。
- 正式权重 `/data/model/DeepSeek-V4-Flash-0731-w8a8`，正式第 2 层 CSA
  输出直接接第 3 层 HCA 输入，使用实际服务 custom-op 入口并断言没有 Native fallback。
- CSA 性能版、HCA 现有服务实现；同一个手动 NPUGraph 包含 CSA→HCA。
  不包含 MoE、EPLB、整模型或 16 卡测试。
- CANN 9.2.0-beta.2、NZ2、atomic_add=0、确定性 level1、HCCL_DETERMINISTIC=true。
- 固定 metadata、合成历史；每次重放恢复同一 cache/state 初态，恢复在计时区间外。
  这是联合重放对照，不是连续生成 token 的三个 decode step。
- 编译、权重加载和 warmup 后，关闭 DFX/Profiler 采 10 次 NPU Event；
  随后另采三个 step 的 PyTorch+DFX。以下两类时间分开呈现。

无 profiler 的联合 replay，单位 μs：

| 配置 | min | mean | max |
| --- | ---: | ---: | ---: |
| 保留 checksum | 222261.15 | 224361.40 | 225649.26 |
| 去除 replay checksum | 205985.11 | 207478.31 | 209156.75 |

三个 profiling step 中的 AICPU 区间，单位 μs：

| 配置/算子 | min | mean | max |
| --- | ---: | ---: | ---: |
| Before CSA | 146300.62 | 146645.77 | 147224.70 |
| After CSA | 138371.86 | 139044.68 | 139632.24 |
| Before HCA | 77684.48 | 78222.75 | 78576.78 |
| After HCA | 69321.32 | 69611.91 | 69854.04 |

同轮泳道的核内任务首尾区间，含任务间空隙、不是各 task 耗时之和，单位 μs：

| 配置/算子 | min | mean | max |
| --- | ---: | ---: | ---: |
| Before CSA | 530.70 | 548.55 | 565.06 |
| After CSA | 520.24 | 540.18 | 553.98 |
| Before HCA | 324.98 | 341.49 | 355.22 |
| After HCA | 331.18 | 339.08 | 348.40 |

核内算术、任务配置和缓存策略均未改。核内区间的小幅差异不能当成新增 kernel 优化。
去除 checksum 后，两个 AICPU 区间均值合计约 208657 μs，核内区间合计仅
879 μs，仍有约 99.6% 在核内首尾区间之外。现有泳道没有给 graph restore
中的 copy/zero/flush、callable bind 单独打点；这些环节的具体占比尚未证实。
下一步若继续定位，应先量化这些 runtime 阶段，不能继续把主要差距归给 incore task。
本轮不替换 Native/TMR 七档性能基线，也不修改默认 runtime。

## 采集修复、变更与正确性

首次在 Simpler `ac1654822` 开启 DFX 时，CSA 第二次调用触发 AIV 507015；
关闭 DFX 的联合重放可以通过。代码定位到 HBG kernel-mode 绕过 program-mode
初始化，只让 AICore 开启采集，却没有给 AICPU 发布记录区和 rotation table。

Simpler `585ef2798` 补齐 AICPU 采集初始化、每次 launch 的边界和每 callable 的
Host 依赖图输出。Before/After 都包含该修复，避免把采集修复混入 checksum 消融。
`ef6812891` 仅在 AICPU replay 将完整 `validate_graph_packet` 改成 framing 检查：
Host 构建/提交仍做完整 FNV 校验，设备的身份、地址/容量、版本、generation 和
镜像语义检查保留；没有跳过 copy、zero、flush、bind。
设备 replay 不再承诺靠 checksum 检出任意 payload bit flip，依赖 Host 验证和
CANN 对不可变副本的复制/生命周期保证。PyPTO `cb2484471` 同步子模块和 ABI pin。

Before、After 的 graph/eager 以及 After/Before 的 12 项输出、Top-K、cache/state
零容差比较全部通过，浮点 max_abs=0、整数 mismatches=0；保护区检查通过。
本轮未重新验证 Native 的既有数值差异或整模型 token/DSpark。
HCA 本轮 history 没有跨 C128 压缩边界，跨边界写入检查见
[前一轮 HBG 精度记录](../hbg_accuracy_20260930/README.md)。

- Before：`task_20260930_212435_28921078924`。
- After：`task_20260930_213111_29304566656`。
- 原始目录：`tests/pypto_test/results/hbg_joint_profile_20260930/{before_crc,after_crc}/`。
  目录中的 crc 是实验路径名，实际移除的是 FNV checksum。
- 初次 DFX 故障保存在 `before/`、`before_level1/`，失败任务不计入耗时统计。
  `before_pytorch/` 是修复前无 DFX 的预采，不与这次正式 A/B 混用。
- 原始 `states.pt` 和编译文件仅保留在 results，未加入 Git。
- Simpler packet/slot/restore 56 项 CPU 测试、PyPTO kernel ABI 46 项通过；
  采集脚本 ruff、Python 编译检查及 shell 语法检查通过。
  全仓 `format.sh ci` 因缺少 pre-commit 未执行完成；系统 clang-format 太旧，
  不支持 Simpler 的 BlockIndent 配置，未宣称 C++ 格式检查通过。

## 复现

安装当前工作区配套的 PyPTO/Simpler 后，在接入仓执行。先整包冻结算子，
再由队列分配单卡；使用逻辑卡号 0：

```bash
repo="$PWD"
result_root="$repo/tests/pypto_test/results/hbg_joint_repro"
mkdir -p "$result_root"
cp -a vllm_ascend/ops/pypto "$result_root/frozen_ops_pypto"
/data/server-toolkits/pto-task/app/task-submit --device auto --max-time 1200 \
  "bash $repo/tests/pypto_test/hbg_joint_profile_20260930/run.sh \
  $result_root/current --source $result_root/frozen_ops_pypto \
  --history 131072 --batch 4 --swimlane-level 4 --device 0"
# 队列任务成功结束后，在公共环境中离线导出泳道：
python tests/pypto_test/hbg_joint_profile_20260930/export.py --input "$result_root/current"
```

需要前后对照时，以 `--reference <before>/states.pt` 加入同初态零容差检查。
