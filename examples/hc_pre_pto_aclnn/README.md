# HC_pre 自定义算子开发与调用示例

本示例将 PyPTO 生成的 HC_pre C++ 核函数封装为 CANN 自定义算子，并通过 PyTorch C++ Extension 提供调用接口。可用于学习如何复用 PTO 生成代码、对接 ACLNN，以及验证独立算子的精度和性能。

调用流程：

```text
Python：torch.ops.hcpre_pto.hc_pre(x, weight, scale, base)
  → Torch 绑定：检查输入、创建输出
  → ACLNN：aclnnHcPrePtoGetWorkspaceSize / aclnnHcPrePto
  → MIX 核入口：按平台 AIC 核数启动，AIC:AIV = 1:2
  → generated/ 中的 PTO C++ 核函数
```

当前支持独立推理调用、Meta 形状推导、NPU Graph 重放，以及下文的 `npugraph_ex` SuperKernel 示例。尚未替换 vLLM 模型中的 HC_pre 调用，不提供反向传播。

## 1. 前提条件

- AArch64 Linux，以及支持 `ascend910_93` 目标、AIC:AIV 核数比为 1:2 的昇腾设备。
- 已安装匹配的驱动、固件、CANN 开发环境和 PyTorch / TorchNPU。
- CMake ≥ 3.20、C++ 编译工具、Ninja，以及 PTO-ISA 源码和头文件。
- Python 测试依赖：NumPy；从模型目录读取权重时另需 safetensors，默认随机权重模式不需要。
- 保留完整 vLLM Ascend 源码目录：Torch 绑定复用 `csrc/aclnn_torch_adapter/`，无需编译整个 vLLM。

已验证的依赖版本：

|依赖|版本或状态|
|---|---|
|Python|3.12|
|PyTorch / TorchNPU|`2.10.0+cpu` / `2.10.0.post2`|
|CANN 编译器及运行时组件|`version.info` 为 `9.2.0 / 20260916`|
|PTO-ISA|`hw-native-sys/pto-isa@327cd5869f3a7c4d2c6a1b945b2aed06e7665c5d`|

这里直接编译已有的生成代码，**构建和运行 PTO 算子均不需要 PyPTO、Simpler 或 Native 对照库**。重新从 Python 算子生成 C++ 属于另一个步骤。

## 2. 示例目录

```text
hc_pre_pto_aclnn/
├── CMakeLists.txt                 # CANN 自定义算子构建入口
├── generated/
│   ├── hc_*.cpp                  # 六档生成的 AIC/AIV 核函数
│   ├── hc_*.pto                  # 对应的 PTO IR
│   └── manifest.json             # 来源提交及文件 SHA256
├── include/contract.h            # 分档、任务划分及工作区大小计算
├── op_host/hc_pre_pto.cpp         # 算子定义、输入约束和 Host tiling
├── op_kernel/hc_pre_pto.cpp       # MIX 入口和生成函数的参数适配
├── torch/binding.cpp             # Torch 算子注册及 ACLNN 调用
├── scripts/
│   ├── build.sh                  # 编译并打包 CANN 算子
│   ├── build_torch.py            # 编译 Torch 扩展库
│   ├── run_torch.sh              # T96 功能测试，可选 Native 对照
│   ├── run_all.sh                # 十四场景测试、采集和结果汇总
│   ├── export_all.py             # 离线解析设备 profiler 数据
│   ├── summarize_all.py          # 生成性能对比文档
│   └── append_repeat.py          # 将复测值追加到已有表格
├── tests/
│   ├── check_outputs.py          # CPU 参考及 T96 输入生成
│   ├── test_torch.py             # Torch 精度、非法输入及图重放检查
│   ├── bench_all.py              # 十四场景对照，默认随机权重
│   └── run_aclnn.cpp             # 直接调用 ACLNN 的 C++ 测试入口
├── reports/                      # 实测数据来源记录
├── build_all/                    # 构建后生成的 CANN 包及解压目录
├── build/torch/                  # 构建后生成的 Torch 扩展库
├── output/                       # 运行后生成的报告和采集数据
├── README.md                     # 本使用说明
└── 全场景性能对比.md              # 历史数据、复测结果及验证详情
```

`build*/`、`output/` 和 profiler 临时目录已被 Git 忽略。它们由构建或测试生成，无需作为源码提交。

## 3. 算子接口

```python
mixed, post, comb = torch.ops.hcpre_pto.hc_pre(x, weight, scale, base)
```

所有输入需为同一 NPU 上的连续 ND Tensor。`T = B × S`，调用前合并 B/S 两个维度。

|参数|数据类型|形状|
|---|---|---|
|x|BF16|`[T, 4, 4096]`|
|weight（hc_fn）|FP32|`[24, 16384]`|
|scale（hc_scale）|FP32|`[3]`|
|base（hc_base）|FP32|`[24]`|
|mixed 输出|BF16|`[T, 4096]`|
|post 输出|FP32|`[T, 4]`|
|comb 输出|FP32|`[T, 4, 4]`|

Host 根据 T 自动选择生成代码，调用者无需指定分档：

|T 范围|生成文件前缀|
|---|---|
|1～32|`hc_small`|
|33～48|`hc_medium48`|
|49～95|`hc_medium96`|
|96|`hc_exact96`|
|97～144|`hc_large144`|
|145 及以上|`hc_large`|

接口检查范围为 `1 ≤ T ≤ 1048576`；这不表示所有大小均已实测。已测场景见性能对比文档。

## 4. 编译算子与 Torch 扩展

激活装有上述 Python 依赖的环境，将下方 `/path/to/...` 替换为实际安装或源码位置。

```bash
source /path/to/cann/set_env.sh
cd /path/to/vllm-ascend/examples/hc_pre_pto_aclnn

# 编译 CANN 自定义算子并生成安装包
bash scripts/build.sh /path/to/pto-isa build_all

# 将安装包解压到独立目录
bash build_all/custom_opp_hce_aarch64.run \
  --noexec --extract=build_all/private_opp

# 编译 Torch 绑定
python scripts/build_torch.py
```

主要产物：

- `build_all/private_opp/packages/vendors/hcpre_pto/`：CANN 算子包，包含 ACLNN 库和设备核。
- `build/torch/hcpre_pto_torch.so`：Python 通过 `torch.ops.load_library` 加载的扩展库。

## 5. 调用样例

在本示例目录中，先为 Python 进程设置算子包及动态库路径：

```bash
export ASCEND_CUSTOM_OPP_PATH="$PWD/build_all/private_opp/packages/vendors/hcpre_pto"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
```

以下代码演示 T96 调用；随机权重仅用于演示接口，性能对照使用第 7 节的正式权重。

```python
from pathlib import Path
import torch
import torch_npu

torch.npu.set_device(0)  # 当前进程可见的第 0 张卡
torch.ops.load_library(str(Path("build/torch/hcpre_pto_torch.so").resolve()))

x = torch.randn(96, 4, 4096, device="npu", dtype=torch.bfloat16)
weight = torch.randn(24, 16384, device="npu", dtype=torch.float32) * 0.01
scale = torch.ones(3, device="npu", dtype=torch.float32)
base = torch.zeros(24, device="npu", dtype=torch.float32)

mixed, post, comb = torch.ops.hcpre_pto.hc_pre(x, weight, scale, base)
torch.npu.synchronize()
print(mixed.shape, post.shape, comb.shape)
```

预期输出形状：

```text
torch.Size([96, 4096]) torch.Size([96, 4]) torch.Size([96, 4, 4])
```

使用共享设备队列时，所有 NPU 调用均需在分配到设备的任务中运行。以下测试脚本使用 `TASK_DEVICE` 设置可见设备。

## 6. 功能验证

先生成 T96 测试输入，此步骤仅使用 CPU：

```bash
python tests/check_outputs.py output/input --generate
```

在已分配设备、已初始化 CANN/Python 环境的任务中执行：

```bash
bash scripts/run_torch.sh \
  build_all/private_opp/packages/vendors/hcpre_pto \
  output/input output/torch_check
```

使用 `task-submit` 的完整提交形式如下。任务内也需要激活 Python 环境；输出目录应为尚不存在的新目录。

```bash
task-submit --device auto --max-time 600 --run '
  source /path/to/cann/set_env.sh
  source /path/to/venv/bin/activate
  cd /path/to/vllm-ascend/examples/hc_pre_pto_aclnn
  bash scripts/run_torch.sh \
    build_all/private_opp/packages/vendors/hcpre_pto \
    output/input output/torch_check
'
```

测试检查三个输出与 CPU 参考、非法输入拒绝、图重放一致性，以及修改输入后重放是否读取新数据。结果保存到指定输出目录的 `report.json`。此步骤不需要模型权重和 Native 对照库。

## 7. 十四场景性能对照（可选）

该测试使用 `B ∈ {2,4,8,16,20,24,28}`、`S ∈ {5,6}`，共 14 组场景。额外需要：

|参数|用途|
|---|---|
|Native vendor 目录|安装好的 AscendC HC_pre 对照算子包，目录名通常为 `packages/vendors/customize`|
|`hc_pre_recipes_reference.so`|Native 的 Torch 绑定库，注册 `torch.ops.custom.npu_hc_pre`|

Native 为 `cann-recipes-infer@4c3d1e1258053c4d810322198a7fd1722c89897e` 的 HC_pre 实现；其算子包和 Torch 绑定需另行准备，未随本示例提供。

默认生成固定种子的随机权重：fn 为标准差 0.01 的正态分布，scale 为全 1，base 为标准差 0.1 的正态分布；两侧共用同一组权重。无需准备模型目录。

在队列分配的任务中，初始化相同的环境后执行：

```bash
bash scripts/run_all.sh \
  build_all/private_opp/packages/vendors/hcpre_pto \
  /path/to/native/packages/vendors/customize \
  /path/to/hc_pre_recipes_reference.so \
  output/all14
```

需要复现 10-12 版本的真实权重条件时，在命令末尾追加可选参数：

```bash
  --checkpoint /path/to/DeepSeek-V4-Flash-0731-w8a8
```

该模式只读取第 2 层 attn 的 fn/scale/base，不加载模型。随机模式可通过 `--weight-seed 20261009` 指定权重种子，默认值为 20261009。输入 x 的种子仍为 `20261009+B*100+S`。

脚本完成采集后，等待执行进程退出，再调用 `msprof` 离线解析。报告写入 `output/all14/report.json`，本轮表格写入 `output/all14/summary.md`，不会覆盖仓内历史性能表。报告记录权重来源、种子和 SHA256；随机权重结果不能当作真实权重基准的同条件复现。

- **核耗时**：读取 `op_summary` 中对应算子的 `Task Duration(us)`，每侧 3 次取中位数。
- **图重放耗时**：图外 NPU Event 包住一次 replay，两侧交替各测 20 次；会受提交间隙影响，不能当作纯核耗时。
- **精度**：BF16 的 atol/rtol 为 `0.01/0.01`，FP32 为 `1e-5/1e-4`，沿用 10-12 版本 benchmark 的回归阈值。

具体数据与检查例外见 [全场景性能对比](全场景性能对比.md)。完整报告保留逐输出误差，应同时检查 CPU 对照和修改输入后的检查结果。

## 8. 接入代码说明

- `op_host/` 决定分档并计算工作区大小；`op_kernel/` 将工作区划分为 parts、RMS 和每个 MIX 核组独立的 FIFO。
- MIX 入口在 AIC 侧调用生成的 `*_aic`，在 AIV 侧调用 `*_aiv`；两侧使用同组 FIFO，由生成代码中的 TPUSH/TPOP/TFREE 完成通信和同步。
- 生成函数中的 `v1、v2…` 是编译器参数名，传参顺序依据同批生成的 kernel wrapper 对齐。更新后的 medium48 与其他动态档均为 weight/x 顺序；exact96 的标量参数数量不同，入口单独适配。
- `torch/binding.cpp` 负责输入检查、输出申请和 `EXEC_NPU_CMD(aclnnHcPrePto, ...)` 调用；配套适配实现来自仓内 `csrc/aclnn_torch_adapter/`。

优化接入基于 ACLNN 分支提交 `cf0ca6ef0d8202bc1ed2ab0656af9e108451e93e`，生成文件对应 Python 优化提交 `8e875aa32be3546dd487057ec9d0ebfdd9a6f48b`。各文件来源与校验值见 [manifest.json](generated/manifest.json)。C++ 生成内容不手改；PTO IR 仅将源码定位路径归一化。

| T 范围 | 更新文件（generated/） | 本次改动 |
|---|---|---|
| 1～32 | 保留 hc_small.cpp / .pto | 无 |
| 33～48 | hc_medium48.cpp / .pto | RMS 输入复用＋Sinkhorn 整数索引 |
| 49～95 | hc_medium96.cpp / .pto | Sinkhorn 整数索引 |
| 96 | 保留 hc_exact96.cpp / .pto | 无 |
| 97～144 | hc_large144.cpp / .pto | Sinkhorn 广播加法 |
| >144 | hc_large.cpp / .pto | Sinkhorn 整数索引＋广播加法 |

T33～48 新生成接口的第三、四个指针为 weight、x，因此入口改用 `COMMON_ARGS`；每核组 FIFO 从 98304 降为 49152 个 FP32 元素，`contract.h` 同步调整。其余参数映射与原入口一致，Early-Start 的 Wait/Set 接入保持原状。

2026-10-16 已分别重新构建、安装原版与优化版，在同一张 Ascend910_9392（24 AIC / 48 AIV）上，通过 C++ 直接调用 `aclnnHcPrePtoGetWorkspaceSize` → `aclnnHcPrePto` 完成验证。18 个兼容用例覆盖六档和 PR 的全部 14 个 B/S 场景，两版各四轮均通过；每轮预热 5 次、统计 20 次，优化版 y/post/comb 的超差元素数均为 0。单元素容差为 `0.005 + 0.005 × |golden|`，y/post、comb 的允许超差比例分别为 0.5%、2%。

按设备 Task Duration 的 80 个稳态样本中位数比较，medium48 的 T=33/40/48 耗时降低约 7.4%/5.1%/5.2%，四组配对均更快。其余改动档位尚未确认稳定收益；未改分档也存在轮间波动。本次是直接 ACLNN 验证，不代表优化生成核已重新完成 Torch、SK 或 Early-Start 验收。使用更新后的生成文件需重新构建安装包，不能复用旧二进制。

### 优化生成核的十四场景性能

原版为 `cf0ca6ef0d8202bc1ed2ab0656af9e108451e93e`，优化版为本次接入的生成核（Python 来源提交 `8e875aa32be3546dd487057ec9d0ebfdd9a6f48b`）。下表为 2026-10-16 同卡 C++ 直接 ACLNN 对照，不经过 Torch、图重放或 SK。

每版每例运行四轮，每轮丢弃 5 次预热，合并四轮共 80 个有效设备 `Task Duration(us)` 取中位数，不是四个中位数的平均值，也不包含 Host 查询和内存分配耗时。合并收益为 `(原版中位数−优化版中位数)/原版中位数`；配对收益范围按每轮各 20 次样本的中位数计算，正数表示优化版更快。执行顺序为原版1→优化版1→优化版2→原版2，再原版3→优化版3→原版4→优化版4。

small、exact96 的生成核未修改，其轮间变化用于提示测量波动；不能将表中小幅变化都归因于优化。medium48 四组配对均为正，其余改动档位尚无稳定收益证据。

| B | S | T | 分档 | 原版 μs | 优化版 μs | 合并收益 | 配对收益最小～最大 |
|---|---|---|---|---|---|---|---|
| 2 | 5 | 10 | small | 21.98 | 22.14 | -0.7% | -14.0%～+11.3% |
| 2 | 6 | 12 | small | 22.20 | 22.29 | -0.4% | -13.5%～+11.6% |
| 4 | 5 | 20 | small | 23.38 | 23.32 | +0.3% | -12.1%～+11.5% |
| 4 | 6 | 24 | small | 22.92 | 23.15 | -1.0% | -13.0%～+10.6% |
| 8 | 5 | 40 | medium48 | 25.01 | 23.73 | +5.1% | +0.8%～+13.8% |
| 8 | 6 | 48 | medium48 | 24.91 | 23.61 | +5.2% | +0.1%～+14.5% |
| 16 | 5 | 80 | medium96 | 23.24 | 22.86 | +1.6% | -4.9%～+11.2% |
| 16 | 6 | 96 | exact96 | 21.89 | 22.06 | -0.8% | -6.2%～+10.1% |
| 20 | 5 | 100 | large144 | 27.04 | 26.77 | +1.0% | -8.3%～+11.5% |
| 20 | 6 | 120 | large144 | 27.74 | 27.46 | +1.0% | -8.3%～+11.5% |
| 24 | 5 | 120 | large144 | 27.75 | 27.45 | +1.1% | -7.8%～+9.6% |
| 24 | 6 | 144 | large144 | 28.92 | 28.83 | +0.3% | -8.9%～+10.9% |
| 28 | 5 | 140 | large144 | 28.33 | 28.55 | -0.8% | -10.3%～+9.8% |
| 28 | 6 | 168 | large | 26.82 | 26.63 | +0.7% | -4.7%～+7.9% |

### 补充边界用例

| T | 分档 | 原版 μs | 优化版 μs | 合并收益 | 配对收益最小～最大 |
|---|---|---|---|---|---|
| 1 | small | 16.72 | 16.81 | -0.5% | -10.0%～+10.5% |
| 33 | medium48 | 25.66 | 23.75 | +7.4% | +0.9%～+11.9% |
| 6 | small | 19.59 | 19.66 | -0.4% | -13.5%～+10.2% |
| 32 | small | 24.00 | 23.35 | +2.7% | -5.4%～+8.9% |

## 9. 通过 npugraph_ex 融合三个算子

2026-10-15 在 Ascend910_9362（20 AIC / 40 AIV）、Python 3.11.4、
Torch `2.12.0+cpu`、TorchNPU `2.12.0+git5462a1b`、CANN 9.2.0 上验证了一个 T96 场景：

```python
class Chain(torch.nn.Module):
    def forward(self, x, weight, scale, base, addend, multiplier):
        torch.npu.super_kernel_scope_begin("hcpre_add_mul")
        shifted = torch.ops.hcpre_pto.ascendc_add(x, addend)
        mixed, post, comb = torch.ops.hcpre_pto.hc_pre(shifted, weight, scale, base)
        scaled = torch.ops.hcpre_pto.ascendc_mul(mixed, multiplier)
        torch.npu.super_kernel_scope_end("hcpre_add_mul")
        return scaled, post, comb

compiled = torch.compile(
    Chain(), backend="npugraph_ex", dynamic=False, fullgraph=True,
    options={"static_kernel_compile": True, "super_kernel_optimize": True,
             "clone_input": False},
)
```

这里前后两个算子是独立的 AscendC Add/Mul，HC_pre 保持原来的 Torch → ACLNN → PTO 调用路径。
本机内置 BF16 `torch.add/torch.mul` 的静态编译选择了 legacy TBE 实现，没有 SuperKernel bind map，
因此仅标记 scope 不能把它们融合进去。`--neighbours torch` 保留这条诊断路径；正式验收默认使用
`ascendc_add/ascendc_mul`，没有覆盖 PyTorch 全局分发，也没有修改系统 CANN。
这两个示例算子仅接受同形状、连续 BF16 输入，元素数为正的 16 倍数，不提供广播和类型提升。

### 接入修改的作用

- **源码完整打包**：在线静态编译读取已安装的源码，需要一起安装 `include/` 和 `generated/`。
  只带入口 CPP 的旧安装包能运行预编译核，但重编译会因缺少 `../include/contract.h` 失败。
- **逻辑核号适配**：`include/pto_superkernel_adapter.h` 将固定 PTO 版本头文件中的原始核号接口
  局部映射到 AscendC 接口。PTO MIX 代码需要核组编号，因此使用
  `AscendC::GetBlockIdx() / AscendC::GetTaskRatio()`；不能直接把线性 AIV 编号当核组编号。
  适配宏在 include 后撤销，不改外部 PTO 源码，也不绕过编译器检查。
- **核数与工作区一致**：Host tiling 传入实际核组数，设备任务步长、sink/worker 分配与 FIFO
  工作区都使用同一个值。生成的六档 CPP/IR 和计算公式保持原样；本次设备验收仅覆盖 T96。
- **可融合的编译产物**：`static_kernel_compile` 与 `super_kernel_optimize` 同时开启，
  CANN 为 AscendC 包装的 PTO 核生成子核信息，再由 SuperKernel 更新捕获图。
  本例保留 PTO 内部同步，融合后仍为完整的 20 AIC / 40 AIV MIX 执行。

### 构建与复现

沿用第 4 节构建方式；编译并行度由 CMake 环境变量控制。本机使用 40：

```bash
export CMAKE_BUILD_PARALLEL_LEVEL=40 MAX_JOBS=40
bash scripts/build.sh /path/to/pto-isa build_chain
python scripts/build_torch.py

# 两组使用同一安装包和 Torch binding，各自在独立进程、独立 OPP 目录运行。
bash scripts/run_superkernel.sh build_chain output/sk_graph graph
bash scripts/run_superkernel.sh build_chain output/sk_fused superkernel
python scripts/compare_superkernel.py output/sk_graph output/sk_fused \
  --output output/sk_comparison.json
```

需要 Python `tqdm` 供 CANN 静态编译工具使用。Profiler 的解析器需要可加载的 `libsqlite3.so`；
本机只有 `libsqlite3.so.0`，在私有库目录增加链接并加入 `LD_LIBRARY_PATH` 后可用。
不需要复制或修改系统 Python 环境。

`run_superkernel.sh` 自动解压指定构建包，建立私有 OPP 视图，在退出时卸载本次生成的静态核包。
CANN 的目录识别要求 `built-in` 本身是实际目录；其下内容使用指向原安装的只读链接。
`op_compiler` 还需要 OPP 旁的 `aarch64-linux/data/platform_config`，脚本也建立对应链接。
输出目录必须不存在，避免复用不匹配的静态编译缓存。

输入固定地址，第二次调用原地更换所有输入；每个输入前后各保留 64 个元素保护区。
`clone_input=False` 保证静态编译和图捕获看到一致的 storage offset。
本机默认克隆会把偏移 64 变成 0，使静态核匹配失败，最终回退到不可融合的普通核。
这不代表框架没有其他数据复制：设备 profile 中仍可见 scope 外的 `ForeachCopy`。

### 验收证据与性能口径

测试同时检查三个输出、有限值、换输入重放、输入未被改写和前后保护区。
数值容差与已有 CPU 参考一致：mixed 使用 `atol=0.01, rtol=0.01`，post/comb 使用
`atol=1e-5, rtol=1e-4`。两组的两个输入种子均通过。

`test_superkernel.py` 还强制检查实际融合证据，静默回退会使测试失败：

- `sk_fused_nodes.log` 只有一个 SK，成员依次为 HcPreAdd、HcPrePto、HcPreMul，共 3 个。
- `sk_mdl_updated.json` 只有一个 `sk_entry_mix12` 设备任务。
- `sk_task_queue.json` 只有一个 scope，AIC 函数数为 1，AIV 函数数为 3。

30 个样本，每样本 8 次调用；下表 min/mean/max 均为每组 8 次的平均单次耗时，单位 μs。
Profiler 阶段包含 240 次完整链，独立于未开启 profiler 的 Event 计时阶段。

|计时范围|未融合 min / mean / max|SuperKernel min / mean / max|均值变化|
|---|---|---|---|
|设备 scope：Add 开始至 Mul 结束，含三核间隔|58.926 / 59.114 / 59.291|41.841 / 42.062 / 42.271|-28.85%|
|设备核耗时之和，不含三核间隔|56.896 / 56.995 / 57.169|41.841 / 42.062 / 42.271|-26.20%|
|完整编译调用 Event 窗口，含框架处理与下发空隙|323.118 / 329.504 / 334.555|329.905 / 336.278 / 346.770|+2.06%|

本次满足 30% 浮动范围。设备 scope 的收益不能直接解释为完整调用加速，完整调用仍受框架开销影响。
结果与精简的融合证据保存在 `reports/superkernel_20261015/`；复现后原始 profile、静态编译日志、
输出张量和完整 `sk_meta` 都在指定输出目录。

上述 2026-10-15 验收没有加入 `WaitPreTaskEnd/SetNextTaskStart`，Early-Start 为关闭状态。
上表收益来自 SuperKernel 融合，不能作为 Scalar 与计算流水提前重叠的证据。

## 10. 可选 Early-Start 接入与三组对照

2026-10-16 在同一台机器、同一依赖环境上增加了 CANN 原生 Early-Start 的局部适配。
这是对第 9 节已经融合的三个算子再做开关对照，仍只验收 T96。默认 `early_start=0`。

### 接口位置和适配边界

- HC_pre 在工作区与标量配置完成后、进入生成函数前调用 `AscendC::WaitPreTaskEnd()`，
  所有生成函数返回后调用 `AscendC::SetNextTaskStart()`。六个 tiling key 的每个参与核均调用一次。
- Add/Mul 的 Wait 位于 `TPipe` 对象构造之前，避免提前初始化设备状态；Set 位于整个计算循环之后，
  包括没有 tile 的核也会执行。原有队列释放、TPipe 析构和同步保持完整。
- 不使用 `FORCE=true`，由 CANN 在线编译和
  `super_kernel_optimize_options={"early_start": 0或1}` 控制。计算量、依赖、核数和计时范围保持一致。
- PTO 内部 MIX `SYNCALL` 使用的同步标记生命周期位于 Wait/Set 之间；未移动内部屏障或修改六份生成核。

接口边界参考 [WaitPreTaskEnd](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/inter_task_sync/WaitPreTaskEnd.html)、
[SetNextTaskStart](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/inter_task_sync/SetNextTaskStart.html)
及 [同步标记占用说明](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/inter_core_sync/key_features.html)，
并核对本机 CANN 9.2.0 的实际头文件实现。

### 复现三组对照

第一组必须使用提交 `eee187d0c` 构建的不含 Wait/Set 的包，不能拿当前源码再次构建冒充原代码。
可在独立 worktree 构建该包；三组共用当前测试脚本与相同 Torch binding，绑定 ABI 未变。

```bash
# 在仓库内创建原代码的独立目录，再用已经配置的 CANN/Python 环境构建。
git worktree add --detach /path/to/hcpre-before eee187d0c
bash /path/to/hcpre-before/examples/hc_pre_pto_aclnn/scripts/build.sh \
  /path/to/pto-isa /path/to/build_no_hooks

# 以下命令在当前 examples/hc_pre_pto_aclnn 目录运行。
bash scripts/build.sh /path/to/pto-isa build_early_start
python scripts/build_torch.py
bash scripts/run_superkernel.sh /path/to/build_no_hooks output/early_no_hooks superkernel 0
bash scripts/run_superkernel.sh build_early_start output/early_off superkernel 0
bash scripts/run_superkernel.sh build_early_start output/early_on superkernel 1
python scripts/compare_early_start.py output/early_no_hooks output/early_off output/early_on \
  --output output/early_comparison.json
```

三组串行运行，输出目录各不相同，均为 30 样本 × 每样本 8 次。
设备耗时来自独立 profiler 阶段的 240 个完整 SK 任务；Event 窗口另行采样，含框架复制与下发空隙。

|代码与开关|设备 SK min / mean / max（μs）|完整调用 Event min / mean / max（μs）|
|---|---|---|
|原代码，无 Wait/Set，关闭 Early-Start|44.428 / 44.657 / 44.991|340.183 / 366.465 / 412.900|
|加入 Wait/Set，关闭 Early-Start|42.833 / 43.050 / 43.228|321.137 / 325.645 / 332.285|
|加入 Wait/Set，开启 Early-Start|43.059 / 43.235 / 43.523|341.145 / 349.642 / 359.572|

同一新包开启相对关闭，设备均值增加 **0.43%**，完整调用窗口增加 **7.37%**；均在 30% 浮动范围内。
本轮没有观察到净性能收益，也不据这一轮的小差异断言稳定回退。
第一、二组之间的差异不能归因于 Early-Start 收益：两组运行时均未开启它，本次没有加入公共运行时优化。

### 功能证据和收益限制

三个模式的两个输入种子、三个输出、输入保持及保护区检查全部通过；各模式输出逐元素完全一致。
开启时三个子算子的 `earlyStartWaitFlag/earlyStartSetFlag` 均为 1，实际入口为
`sk_entry_mix12_early_start`。运行时执行了 Early-Start 同步改写，三个任务的 `funcConfig`
分别为 `0x10 / 0x71 / 0xa0`；关闭时使用普通入口且能力标记均为 0。
测试还拒绝“请求开启但元数据仍关闭”及相反的情况。

这些证据确认原生 Early-Start 已接通，但不等于已经测到 Scalar 与计算流水重叠：

- `generated/hc_exact96.cpp` 的 AIC/AIV 尾部都调用 `ptoas_auto_sync_tail(kBarrierAll)`，
  其中执行 `pipe_barrier(PIPE_ALL)`；本版 HC_pre 的 Set 在这些排空操作之后。
- Add/Mul 保留原有 `TPipe::DestroyWithoutPipeAll()` 对空闲队列事件的等待；即使 SK 析构不再额外
  插入 `PIPE_ALL`，这些等待仍会限制退出时未完成的搬运。Wait 前也只保留了很短的入口配置。

因此本版作为默认关闭的功能示例保留，没有把它作为性能优化默认开启。若继续追求重叠收益，
需要先逐条分析生成核末次搬运、事件回收及跨核同步的生命周期，再决定接口是否能内移；本轮没有叠加候选。

完整样本、运行时同步改写、任务队列和尝试记录见 `reports/early_start_20261016/`。
首轮原代码与开启组均因新增审计脚本的日志匹配问题在计时前退出，修正后分别重跑；
这些尝试的输出检查通过，但未作为性能样本使用。最终增强的同步配置审计又对三组原始日志做了离线验证。
