# 独立 HC_pre：范围、调用与单卡对照

## 计算范围

对齐 [cann-recipes-infer 的 HcPre 融合算子说明](https://gitcode.com/cann/cann-recipes-infer/blob/master/docs/models/deepseek_v4/deepseek_v4_mHC_guide.md)，
当前 Native 基线改为从指定 cann-recipes-infer 源码单独构建的 `custom::npu_hc_pre`，
它直接调用融合 `aclnnHcPre`，关闭可选 `pre_mix` 输入及 `pre` 输出，保持三输出范围。
文末历史数据使用当时 release 包的 `npu_hc_pre_v2`，新旧基线分别标识。
详见 [30 μs 优化过程](HC_PRE_OPTIMIZATION_LOG.md)。

独立入口：`vllm_ascend/ops/pypto/hc_pre.py::hc_pre`。
HC_pre 的完整计算和配置常量均在此文件，运行时只导入 `pypto.language`，
不引用 CSA/HCA 实现或它们的配置模块。计算从本仓库 `deepseek_v4_flash_csa/hc_pre.py` 搬入，
原始算法来自 pypto-lib 的同名实现；当前独立版本已调整分块、归约顺序与任务划分，
并按固定容差通过十四场景，详见优化日志。

1. 将 BF16 残差无损加宽为 FP32，在算子内部完成，计时包含该步骤。
2. 对每个 token 的全部 `hc × D` 元素计算 RMS 倒数。
3. 计算 `x_flat @ hc_fn.T`，乘 RMS 倒数，再按分支施加 scale 和 base。
4. pre 为 `sigmoid + hc_eps`，post 为 `2 × sigmoid`。
5. comb 为四行 softmax 加 `hc_eps`，再执行配置规定的 Sinkhorn 归一化。
6. 用 pre 对四路残差加权求和，舍入为 BF16 mixed；同时输出 FP32 post 和 comb。

本文件有一个动态 `@pl.jit` 入口 `hc_pre`，完整计算在同文件的
`@pl.jit.inline` 函数 `_hc_pre_variant` 中，按编译期分块参数展开。
编排读取输入 tensor 描述符中的 T=B×S，每次只执行一个分支，不回读 device tensor 内容。
reshape 仅构造视图，不增加搬运算子。不同 B/S 若具有同一 T，采用同一分块，仍分别验收。

| 动态 T 范围 | 投影 M/K 分块 | K 分段数 | 跨核槽数 | 泳道任务前缀 |
| --- | --- | ---: | ---: | --- |
| 1～32 | 32/1536 | 11 | 1 | `hc_small` |
| 33～48 | 48/1024 | 16 | 2 | `hc_medium48` |
| 49～96 | 96/512 | 32 | 2 | `hc_medium96` |
| 大于 96 | 32/1536 | 11 | 1 | `hc_large` |

一个覆盖可用组核的 SPMD 完成 CV 投影与分段平方和，经 MIX 硬件屏障后，
两个 AIV 分别计算 Sinkhorn、合并的 pre/post 门控与残差混合。
每次只有所选分支的一对 AIC/AIV 任务，计算范围和同步均包含在 incore 首尾中。
报告记录分支与分块；测试同时核对完整编译任务表以及每次实际执行的分支，防止漏阶段或误执行其他分支。

验证采用 `cv_shape_dispatch_v3_all14` 全量十四场景与 `cv_shape_dispatch_v4_large`
四个受影响场景的增量检查，均通过原数值门禁；未改分支的生成计算体已比对一致。
这些是不同任务的采样，不能混称为同一轮十四场景性能结果。
前四个场景在全量轮中三次均低于 30 μs；最后增量轮 T168 为 39.96 μs，完整性能目标尚未达成。
数据位置和逐次结果见 [优化过程](HC_PRE_OPTIMIZATION_LOG.md)。

七档 BS × 两档 seqlen 的 AscendC profiling 与 PTO L1/L4 泳道已归档到
[十四场景对照目录](results/hc_pre_profiles_20261010/README.md)，每份包含三个 step。
候选源码、成功及失败报告的索引见 [优化证据清单](results/hc_pre_opt_20261009/README.md)。

**HC 内部的 RMS 统计包含在范围内；mixed 后的 input_layernorm 不包含。**
Attention、FFN、HC_post、KV cache 和 metadata 均不在此独立入口中。
当前验证范围为 D=4096、hc=4、Sinkhorn 20 次的十四个 Decode 小 token 场景，未验收 Prefill 大 token 性能。

接口张量均位于设备，输出由调用者预分配：

| 参数 | shape | dtype | 角色 |
| --- | --- | --- | --- |
| x | `[T,4,4096]` | BF16 | 输入残差 |
| hc_fn | `[24,16384]` | FP32 | 正式权重 |
| hc_scale | `[3]` | FP32 | 正式权重 |
| hc_base | `[24]` | FP32 | 正式权重 |
| mixed | `[T,4096]` | BF16 | 输出 |
| post | `[T,4]` | FP32 | 输出 |
| comb | `[T,4,4]` | FP32 | 输出 |

## 调用方式

环境已初始化且张量已在 NPU 上时：

```python
import pypto.torch
from vllm_ascend.ops.pypto.hc_pre import hc_pre

pypto.torch.init(device=0, platform="a2a3", runtime="tensormap_and_ringbuffer")
op = pypto.torch.register(hc_pre, "example_hc_pre::forward")
# mixed/post/comb 由调用者预分配，形状和类型见上表。
op(x, hc_fn, hc_scale, hc_base, mixed, post, comb)
```

这一步只注册和运行独立算子，不安装整模型 forward 替换。

## 编译与性能用例

从仓库根目录运行。先 CPU lowering 或编译，不占卡：

```bash
source ../env-dsv4-0251rc1.sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python tests/pypto_test/dsv4_hc_pre_bench.py \
  --build-only --output tests/pypto_test/results/hc_pre_build
```

把 `--build-only` 换为 `--lower-only` 则只生成 IR。真机统一走队列：

```bash
# 当前独立上游包和绑定的本地构建位置；其他机器按下一节重新构建。
hc_pre_recipe_root="$(realpath ../.cache/hc-pre-recipes-20261009)"
hc_pre_library="$hc_pre_recipe_root/torch-hc-pre-build/hc_pre_recipes_reference.so"
hc_pre_opp="$hc_pre_recipe_root/ops/ascendc/build/_CPack_Packages/Linux/External/CANN-custom_ops-none-linux.aarch64.run/packages/vendors/customize"
task-submit --device auto --max-time 900 \
  "bash $PWD/tests/pypto_test/run_hc_pre_bench.sh $PWD/tests/pypto_test/results/hc_pre_decode --batch-size 2 4 8 16 20 24 28 --seqlen 5 6 --timing incore --profile --native-library $hc_pre_library --native-opp $hc_pre_opp"
```

默认 `--native-backend recipes`，基线文件缺失会报错，不能静默使用旧包。
`--native-backend release` 仅用于复核历史结果。
`--kernel-file /绝对路径/hc_pre候选.py` 可选择完整独立源码快照，供优化试验隔离使用。
`--swimlane-level 4` 补采完整调度与编排泳道；默认仍为 L1。文件名和报告写入实际采集档位，
并从原始记录核对档位。L1/L4 分开执行和统计，L4 额外记录可能影响耗时。

### 构建指定 AscendC 基线

本机源码已经下载到上述目录，版本及差异见优化过程文档。构建和测试绑定均不占用 NPU：

```bash
source ../env-dsv4-0251rc1.sh
hc_pre_repo_root="$PWD"
hc_pre_recipe_root="$(realpath ../.cache/hc-pre-recipes-20261009)"
(
  cd "$hc_pre_recipe_root/ops/ascendc"
  OPS_CPU_NUMBER=16 bash build.sh -n hc_pre -c ascend910_93
)
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python "$hc_pre_repo_root/tests/pypto_test/hc_pre_native_reference/build.py" \
  --recipe-root "$hc_pre_recipe_root" --output "$hc_pre_recipe_root/torch-hc-pre-build"
```

绑定直接编译上游 `npu_hc_pre.cpp` 和 `ops_common.cpp`，本仓库只提供 schema 注册；不改算子源码。
使用构建目录下的独立 vendor 包，不执行共享环境安装。每次报告记录源码 revision 和实际加载 API 路径。

默认读取 `/data/model/DeepSeek-V4-Flash-0731-w8a8` 中第 2 层的三份 `hc_attn_*` FP32 权重。
`--layer` 选择层，`--branch ffn` 选择 HC FFN 权重；不加载整模型。
输入为合成 BF16 残差，每个场景的 CPU 随机种子为 `seed + B×100 + S`，拆开提交也可复现。
只使用正式权重，并不代表使用了真实模型激活快照。

默认覆盖用户指定的 14 个场景，不按 T 去重：

| batch_size | seqlen=5 对应 T | seqlen=6 对应 T |
| --- | ---: | ---: |
| 2 | 10 | 12 |
| 4 | 20 | 24 |
| 8 | 40 | 48 |
| 16 | 80 | 96 |
| 20 | 100 | 120 |
| 24 | 120 | 144 |
| 28 | 140 | 168 |

B20/S6 与 B24/S5 虽然都是 T=120，仍分别构造和采集。
HC_pre 不读取历史 KV，历史长度不是本次变化参数。

每组先检查 Native/PTO 的三份输出，再检查各自 graph/eager 严格一致。
**默认使用 `--timing incore`，每侧预热 5 次、采集 3 个 step：**

- Native：PyTorch profiler 中三个 `HcPre` 设备事件的 duration。
- PTO：L1 泳道，仅取 Worker View 的 `kernel-duration-us` 记录，按 `launch_epoch` 分开三个 replay。
  Worker 条形起点包含 local_setup；真正 incore 起点为 `ts + local_setup_us`，终点为 `ts + dur`。
- 主指标 `span_us`：所有 incore 最早开始到最晚结束，不计首任务开始之前和末任务结束之后的 AICPU 区间。
- 辅助指标 `active_union_us`：所有 incore 区间并集，重叠只计一次；`no_incore_gap_us` 为跨度中没有任何 incore 执行的时间。
  空隙可能含派发、依赖等原因，不把它全部归因于 AICPU。incore 内部等待仍在 kernel 时长中。
- 逐任务保留首尾、块数和核时间总和。核时间总和包含并行核，不作为算子耗时，也不把各任务跨度相加。

PTO L1 与可选 PyTorch profiler 在同一个三次重放窗口采集。上面的指标是采集开启时的测量结果。
JIT 编译、权重读取、输出分配、CPU 比较及 profiler 导出不在这些设备区间内。
`--profile` 额外保留 PTO 外部 PyTorch trace；Native trace 和 PTO 泳道在 incore 模式下始终采集。
命名后的 trace 位于 `traces/Bxx_Sy_Tzzz_{Native_PyTorch,PTO_L1,PTO_PyTorch}_3steps.json`。

历史完整图口径保留为 `--timing replay`：NPU Event 在图外记录，默认 20 个样本，含整次图重放。
该结果放入 `replay_timing`，与默认 `incore_timing` 分开，不拿整图时间充当 incore 时间。
另做同址换输入、输出 NaN 毒化及 PTO 输出前后保护行检查。
跨实现的 BF16 mixed 门禁为 `atol=0.01, rtol=0.01`，FP32 post/comb 为 `atol=1e-5, rtol=1e-4`；
逐项报告最大绝对差、RMSE、零容差差异和超差数。该门禁不等于整模型 token 一致性验收。

`report.json` 保存 B/S/T、配置、数值与全部三次 incore 样本，给出各指标中位数和 Native/PTO 比值。
`profile/Bxx_Sy_Tzzz/{native,pto}/` 保存原始 PyTorch profiler 文件。
每次使用新结果目录，失败也保留报告，不覆盖历史记录。

## 本地调试扩展

2026-10-09 检查时，共享 editable 的 Python 源码含 `system.read_clock`，但安装扩展仍是 9 月 30 日版本，
直接导入报 `Operator 'system.read_clock' not found in registry`。
本用例提供 `--pypto-core /绝对路径/pypto_core*.so`，可选择**同一源码仓库已编译**的调试扩展。
仅修改当前进程的导入映射，不修改 PyPTO/Simpler 源码、共享安装或其他会话环境。
编译命令和真机命令使用同一个扩展；共享安装版本一致时无需此选项。

当前本地路径：`../pypto/build/python/bindings/pypto_core.cpython-310-aarch64-linux-gnu.so`。
真机 shell 会进入输出目录，应向 `--pypto-core` 传绝对路径。

## 2026-10-09 初次验证结果（历史完整图口径）

本节是用户明确 B/S 矩阵与 incore 计时口径之前的记录，不作为当前 14 档的比较结果。
这组历史测量使用最初引用 CSA 计算的入口；源码完整搬入后的验证单独记录在文末。

CPU lowering 和 PTOAS/设备代码编译通过，均使用上述同仓库调试扩展。
任务 `task_20261009_212113_39961297584`（T=24）与
`task_20261009_212213_403559928607`（其余六档）均 exit=0。
七档的三份输出均在固定容差内；Native/PTO 各自 graph/eager 严格一致，同址换输入和输出保护行通过。

| T | Native p50 / μs | PyPTO p50 / μs | Native / PyPTO |
| --- | ---: | ---: | ---: |
| 1 | 50.29 | 111.54 | 0.451 |
| 6 | 45.03 | 109.96 | 0.410 |
| 24 | 53.62 | 117.76 | 0.455 |
| 48 | 60.06 | 123.47 | 0.486 |
| 96 | 75.94 | 139.45 | 0.545 |
| 144 | 77.04 | 150.11 | 0.513 |
| 192 | 86.31 | 162.50 | 0.531 |

这是建立独立执行与测量条件的首版，PyPTO 尚慢于 Native，未进行性能优化。
mixed 最大绝对差最高为 0.015625；FP32 post 最高约 1.824e-5、comb 最高约 1.228e-5。
不能把容差通过描述为逐 bit 一致，也没有整模型输出 token 一致性的证据。
这里测量的是本机图重放区间，不能直接与指南中的历史纯算子耗时作性能比。

本地证据：

- `results/hc_pre_20261009/summary.json`：汇总七档、任务与 trace 路径。
- `results/hc_pre_20261009/{smoke_v1,decode_v1}/report.json`：全部数值指标和原始计时样本。
- `results/hc_pre_20261009/decode_v1/profile/`：除 T=24 外六档、每侧三个 step，共 12 份 trace。

十二份 trace 均核对到三个 ProfilerStep，以及 Native 三次 HcPre 或 PTO 三次设备 worker 执行。
CANN profiler 报 ACL→NPU flow 连线生成失败，因此这些 trace 不用于证明完整的 Host/device 连线。
该 profiler 阶段位于性能采样之后，不进入上表统计。

## 用户指定 14 场景与 incore 口径结果

本节保留完整搬入前、引用 CSA 计算时采集的 14 场景基线，不将旧 trace 标记为新实现的采样。

B=2/4/8/16/20/24/28 × S=5/6 全部通过三输出容差、两侧各自 graph/eager 严格一致、
同址换输入与 PTO 输出保护行。CPU 计时边界测试 6 passed，覆盖并行重叠、local_setup、AICPU 排除、
多 replay 分组与不完整记录拒绝。

任务 `task_20261009_214435_65284432152`、`task_20261009_214525_68514626895`、
`task_20261009_214527_6860216646` 均 exit=0，全部使用 L1 泳道，正式权重不变。
后两个任务分在两张卡并行执行单卡用例；不是双卡协同算子。

下表为各自三个 step 的中位数，单位 μs。首尾与并集分别取中位数。

| B | S | T | Native HcPre | PTO incore 首尾 | PTO 执行区间并集 |
| --- | --- | --- | ---: | ---: | ---: |
| 2 | 5 | 10 | 36.82 | 34.18 | 30.12 |
| 2 | 6 | 12 | 34.32 | 34.26 | 30.88 |
| 4 | 5 | 20 | 36.66 | 43.06 | 37.82 |
| 4 | 6 | 24 | 37.52 | 40.20 | 33.88 |
| 8 | 5 | 40 | 47.46 | 47.36 | 41.58 |
| 8 | 6 | 48 | 46.72 | 47.68 | 42.06 |
| 16 | 5 | 80 | 55.60 | 56.96 | 51.60 |
| 16 | 6 | 96 | 54.40 | 63.52 | 52.42 |
| 20 | 5 | 100 | 59.82 | 65.54 | 59.80 |
| 20 | 6 | 120 | 61.74 | 66.74 | 58.66 |
| 24 | 5 | 120 | 65.24 | 67.38 | 61.60 |
| 24 | 6 | 144 | 63.96 | 73.50 | 66.64 |
| 28 | 5 | 140 | 66.22 | 79.64 | 73.08 |
| 28 | 6 | 168 | 70.06 | 84.42 | 74.98 |

证据目录 `results/hc_pre_incore_20261009/`：

- `RESULTS.md`、`summary.csv`、`summary.json`：14 档比较、任务和文件来源。
- `traces/`：42 份重命名 JSON，每场景 Native PyTorch、PTO PyTorch、PTO L1 泳道各一份。
- `b2_s5_v1/`、`s5_v1/`、`s6_v1/`：原始报告、DFX records、依赖与 PyTorch profiler 文件。

全部 trace 已核对三个 step/launch epoch，以及每次完整七阶段 incore 记录；
Native 每份三次设备 HcPre，PTO 每份三次设备 worker。
PyTorch 与 PTO 原始时钟各自保留，未人为对齐；仍存在前述 ACL→NPU flow 连线缺失。
排除调用外层 AICPU 后，大 batch 的 PTO 计算跨度仍有优化空间；没有改 HC 算术或扩大容差。

## 完整搬入后的独立实现验证

按用户要求，将 RMS、投影、门控、Sinkhorn、四路残差混合及尾块处理完整搬入
`vllm_ascend/ops/pypto/hc_pre.py`，所有模型与分块常量在本文件定义。
此前引用 CSA 实现的形式已删除；CSA/HCA 原文件、整模型 forward 和依赖仓库未改。

对三个搬入函数的参数和计算体做 AST 直接比较，忽略函数名、装饰器、文档字符串与位置信息，
均一致；独立文件的唯一导入为 `import pypto.language as pl`。这项比较证明搬运内容保持一致，
不替代编译或真机验收，也不代表与 Native 逐 bit 相同。
CPU 完整设备代码编译通过，报告为 `build_v1/report.json`。

为验证搬入后的运行边界，重测最小 B2/S5 与最大 B28/S6，包含非整块尾行；未重跑整套 14 场景。
任务 `task_20261009_215759_11422676160`、`task_20261009_215800_114338827564` 均 exit=0。
两组的三份输出均通过原容差、各自 graph/eager 严格一致、同址换输入重算及输出保护行检查。
mixed 最大绝对差均为 0.015625，post/comb 的最大绝对差最高分别约 1.117e-5 / 1.130e-5。

每组仍预热 5 次、记录 3 个 step，以下仅为本次定向验证附带的采样，不宣称性能优化：

| B | S | T | Native HcPre / μs | PTO incore 首尾 / μs | PTO 执行并集 / μs |
| --- | --- | --- | ---: | ---: | ---: |
| 2 | 5 | 10 | 36.36 | 34.02 | 30.40 |
| 28 | 6 | 168 | 72.04 | 83.16 | 73.44 |

证据目录 `results/hc_pre_standalone_copy_20261009/`：

- `migration_comparison.json`：搬入函数计算体比较及独立导入记录。
- `build_v1/report.json`：CPU 完整编译报告。
- `b2_s5_v1/report.json`、`b28_s6_v1/report.json`：真机数值、图重放及三次采样。
- 两个真机目录的 `traces/`：共六份 Native PyTorch、PTO PyTorch 和 PTO L1 JSON。

旧 14 场景结果与本次搬入验证分别保留；提交可读报告和 trace JSON，编译二进制与原始大体积采集文件本地保留。
