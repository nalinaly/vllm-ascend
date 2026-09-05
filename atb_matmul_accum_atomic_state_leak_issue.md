# [Bug] `pp_matmul_accum_atomic` 未关闭 AI Core atomic-add 状态，导致后续 AIC store 被错误累加

## 问题概述

调用 `torch_npu._npu_matmul_add_fp32()` 后，ATB 使用过的 AI Core 上可能仍然
保持 DMA atomic-add 状态。后续一个完全无关的 AIC kernel 即使执行的是普通
L0C-to-GM store，也会变成向 GM 原子累加，而不是覆盖写入。

这是一个跨算子的静默状态污染问题。在下面的最小复现中，普通 store 本应写入
`64`，但受影响的 AIC 实际写入了 `100 + 64 = 164`。在 Qwen3-14B 的 PyPTO
prefill 场景中，同一问题会将原本正常的 logits 变成全 `NaN`。

`torch.npu.synchronize()` 无法解决此问题：同步只能保证任务完成及执行顺序，
不会恢复每个 AI Core 上的 atomic 状态。

## 复现环境

本问题于 2026-08-14 在以下环境中稳定复现：

| 组件 | 版本 |
| --- | --- |
| 设备 | `Ascend910_9362`，20 个 cube core，40 个 vector core |
| 驱动 / `npu-smi` | `25.5.5` |
| CANN | `9.1.0`，内部版本 `V100R001C25B114` |
| ATB | `9.1.0.B150` |
| Python | `3.11.4` |
| PyTorch | `2.12.0+cpu` |
| torch_npu | `2.12.0+git5462a1b`（`5462a1b3352e91893fdf65994b45a4b613b24861`） |
| 用于编写探测 kernel 的 PyPTO | `f3b6e7f0c916435f710bc195a6113168d701cf80` |

截至复现时，ATB 公开 `master` 分支的
`578daaec1e5cf238ecfa8607a24ccbb606b46b6a` 中仍存在相同的状态清理缺失。

## vLLM Ascend 是如何在首个请求前调用到该算子的

这个 ATB 算子不是由模型代码或用户请求直接调用的，而是由 vllm_ascend 的启动
预热逻辑调用。正常的 vLLM V1 engine 初始化路径如下：

```text
vLLM EngineCore._initialize_kv_caches()
  -> model_executor.compile_or_warm_up_model()
    -> Executor.collective_rpc("compile_or_warm_up_model")
      -> 每个 vllm_ascend NPUWorker.compile_or_warm_up_model()
        -> NPUWorker._warm_up_atb()
          -> torch_npu._npu_matmul_add_fp32(x, weight, c)
            -> torch_npu LinearAtb.cpp: enAccum = true
              -> ATB MATMUL_ACCUM_ATOMIC
                -> pp_matmul_accum_atomic
```

### 1. vLLM 在 KV cache 初始化阶段发起 worker 预热

vLLM 的 [`EngineCore._initialize_kv_caches()`][vllm-engine-init] 先初始化 KV
cache，然后在一般启动路径中调用：

```python
self.model_executor.initialize_from_config(kv_cache_configs)
if not envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
    self.model_executor.compile_or_warm_up_model()
```

[`Executor.compile_or_warm_up_model()`][vllm-executor-warmup] 再通过 collective
RPC 将同名方法发送给每个 worker：

```python
compilation_times = self.collective_rpc("compile_or_warm_up_model")
```

因此，TP 场景中的每个 vllm_ascend worker 都会在自己的 NPU 上执行这段预热。

### 2. vllm_ascend 在 worker 预热末尾显式调用 ATB

vllm_ascend 的
[`NPUWorker.compile_or_warm_up_model()`][vllm-ascend-atb-warmup] 先进行模型 dummy
run 和 NPU graph capture，随后对非 A5 设备执行：

```python
# Call ATB matmul to warm up; otherwise, the first operation
# (ReshapeAndCache) may cause performance degradation at runtime.
if get_ascend_device_type() != AscendDeviceType.A5:
    self._warm_up_atb()
```

`_warm_up_atb()` 的完整实现就是：

```python
def _warm_up_atb(self):
    x = torch.rand((2, 4), dtype=torch.float16).npu()
    weight = torch.rand((2, 4), dtype=torch.float16).npu()
    c = torch.rand((4, 4), dtype=torch.float32).npu()
    torch_npu._npu_matmul_add_fp32(x, weight, c)
```

这里返回的 `c` 不会被模型使用；调用目的只是提前加载或预热 ATB，避免首次
`ReshapeAndCache` 的性能下降。但该调用会进入本文后面描述的
`MATMUL_ACCUM_ATOMIC` kernel，并将部分 AIC 留在 atomic-add 模式。

### 3. 遗留状态随后污染第一个真实 PyPTO 请求

`NPUWorker.compile_or_warm_up_model()` 返回后，engine 才开始接收和调度真实请求。
Python 函数返回、stream 任务完成以及 `torch.npu.synchronize()` 都不会重置物理
AI Core 上的配置状态。因此，当首次 PyPTO fused prefill 的普通 L0C-to-GM
store 被调度到受影响的 AIC 时，它继承了预热 kernel 打开的 atomic-add 模式。

这也解释了为什么问题表现为“vLLM 启动正常，但第一个真实请求的 logits 全部
为 `NaN`”：触发 ATB 的代码和出错的 PyPTO kernel 并不在同一个 Python 调用
栈中，中间隔着 worker 启动完成和请求调度，但二者复用了同一设备上的 AIC。

### 4. 当前 vllm_ascend 临时规避

作为 A/B 验证和临时规避，我们在 PyPTO Qwen3 架构上跳过了上述 ATB warmup，
对应提交为 [`b2fc1ecb`][vllm-ascend-workaround]。跳过后，同一模型、权重、输入
和 KV cache 的 logits 恢复正常。该改动只能绕开触发路径，不是 ATB 的通用
修复；其他调用方直接执行 `_npu_matmul_add_fp32` 时仍可能遇到相同污染。

## 最小复现

下面使用 PyPTO 只是为了方便启动一个很小的普通 AIC cube kernel。也可以将其
替换成任意自定义 AIC kernel，只要该 kernel 将 FP32 L0C 结果普通写入 GM，且
在写入前不主动修改 atomic 状态即可。

测试设备有 20 个 cube core，因此代码中使用 `AIC_COUNT = 20`。在其他设备上
复现时，请将其改为对应设备的 cube-core 数量，以确保探测任务覆盖所有 AIC。

```python
import torch
import torch_npu

import pypto.language as pl
from pypto.runtime import RunConfig


AIC_COUNT = 20


@pl.jit
def clear_atomic_state(
    a: pl.Tensor,
    b: pl.Tensor,
    out: pl.InOut[pl.Tensor],
):
    # PyPTO 的显式 atomic store 会生成 SetAtomicAdd，完成 store 后再生成
    # SetAtomicNone。每个 AIC 启动一个 SPMD block，先清除之前测试或进程
    # 可能遗留的 atomic 状态，保证基线对照可靠。
    for block in pl.spmd(AIC_COUNT):
        a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        a_l0a = pl.move(a_l1, target_memory=pl.MemorySpace.Left)
        b_l0b = pl.move(b_l1, target_memory=pl.MemorySpace.Right)
        acc_l0c = pl.matmul(a_l0a, b_l0b)
        pl.store(
            acc_l0c,
            [block * 64, 0],
            out,
            atomic=pl.AtomicType.Add,
        )
    return out


@pl.jit
def ordinary_store(
    a: pl.Tensor,
    b: pl.Tensor,
    out: pl.InOut[pl.Tensor],
):
    for block in pl.spmd(AIC_COUNT):
        a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        a_l0a = pl.move(a_l1, target_memory=pl.MemorySpace.Left)
        b_l0b = pl.move(b_l1, target_memory=pl.MemorySpace.Right)
        acc_l0c = pl.matmul(a_l0a, b_l0b)
        # 默认 store 语义应当是覆盖写入，而不是 atomic add。
        pl.store(acc_l0c, [block * 64, 0], out)
    return out


def reset_all_aic(config):
    a = torch.ones((64, 64), dtype=torch.float16)
    b = torch.ones((64, 64), dtype=torch.float16)
    scratch = torch.zeros((AIC_COUNT * 64, 64), dtype=torch.float32)
    clear_atomic_state(a, b, scratch, config=config)


def run_ordinary_store(config):
    a = torch.ones((64, 64), dtype=torch.float16)
    b = torch.ones((64, 64), dtype=torch.float16)
    # 普通 store 必须用 64 覆盖初始值 100。
    out = torch.full((AIC_COUNT * 64, 64), 100.0, dtype=torch.float32)
    ordinary_store(a, b, out, config=config)
    return out


def run_atb_accum():
    x = torch.ones((2, 4), dtype=torch.float16, device="npu:0")
    weight = torch.ones((2, 4), dtype=torch.float16, device="npu:0")
    accum = torch.zeros((4, 4), dtype=torch.float32, device="npu:0")
    torch_npu._npu_matmul_add_fp32(x, weight, accum)

    # 特意在此同步，用于证明 synchronize 不是有效的修复方式。
    torch.npu.synchronize()
    return accum.cpu()


config = RunConfig(platform="a2a3", device_id=0)

reset_all_aic(config)
baseline = run_ordinary_store(config)
atb_result = run_atb_accum()
after_atb = run_ordinary_store(config)

per_block = after_atb.reshape(AIC_COUNT, 64, 64)[:, 0, 0]

print("ATB 结果的唯一值：", torch.unique(atb_result).tolist())
print("基线结果的唯一值：", torch.unique(baseline).tolist())
print("ATB 后结果的唯一值：", torch.unique(after_atb).tolist())
print("每个 block 的首元素：", per_block.tolist())

assert torch.all(atb_result == 2)
assert torch.all(baseline == 64)
assert torch.any(per_block == 164)

# 清理设备状态，避免影响后续测试。
reset_all_aic(config)
```

运行方式示例：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 python repro_atb_atomic_state.py
```

某次实机运行的输出如下：

```text
ATB 结果的唯一值： [2.0]
基线结果的唯一值： [64.0]
ATB 后结果的唯一值： [64.0, 164.0]
每个 block 的首元素： [..., 64.0, 164.0]
```

具体哪个 block 受影响取决于 AIC 调度，因此不应断言固定的 block 下标。关键
现象是：ATB 调用之后，至少一个普通 store 产生了 `164`。

## 为什么 `164` 可以直接证明 atomic 状态泄漏

探测 kernel 的两个输入都是全 `1` 的 `64 x 64` 矩阵，因此 matmul 结果严格为
`64`。输出 GM tensor 在 kernel 启动前被初始化为 `100`。

- 正确的普通 store 语义：`out = 64`。
- 泄漏后的 atomic-add 语义：`out += 64`，因此 `out = 164`。

ATB 算子自身的结果是正确的，所有元素都为 `2`。因此这不是
`_npu_matmul_add_fp32` 内部的 matmul 精度问题，而是该算子返回后留给下一个
kernel 的设备状态错误。

## 具体代码路径和 bug 点

### 1. torch_npu 请求 ATB 执行累加 matmul

[`LinearAtb.cpp`][torch-npu-linear] 中硬编码了：

```cpp
linearParam.enAccum = true;
```

同时，`C` 被作为第三个输入和输出传给 ATB，从而实现
`C += transpose(x) @ weight`。

### 2. ATB 将其映射到 atomic accumulation kernel

[`LinearOpsRunner::SetupKernelGraphMatmulAccum()`][atb-linear-runner] 将 matmul
类型设置为 `MATMUL_ACCUM_ATOMIC`。

[`matmul_operation.cpp`][atb-matmul-operation] 再将该类型映射到
`PpMatmulAccumAtomicKernel`，其实现位于
[`pp_matmul_accum_atomic.cce`][atb-atomic-kernel]。

### 3. kernel 打开 atomic add，却没有关闭

该 kernel 的入口中包含：

```cpp
AscendC::SetAtomicAdd<float>();
```

之后代码执行选中的 matmul 分支，并直接到达 kernel 末尾。完整文件中只有一个
`SetAtomicAdd`，没有任何 `SetAtomicNone` 或 `DisableDmaAtomic`。

因此，执行过该 kernel 的 AIC 可能一直保留 atomic-add 状态。之后调度到这些
AIC 上的 L0C-to-GM store 会继承累加语义。

## 为什么同步不能修复

`SetAtomicAdd<T>()` 配置的是 AI Core 后续 VECOUT/L0C/L1-to-GM 搬运的行为。
它并不是一条只针对某个地址生效的原子指令：地址仍由具体 store 提供，而该
配置状态决定 store 是直接覆盖，还是进行 atomic read-modify-write。

CANN 官方文档明确建议在 atomic 搬运结束后关闭 DMA atomic 状态，以免影响
后续指令。等待 stream 只能保证之前的任务已经完成，并不会主动发出状态清理
指令。

## 真实模型中的 A/B 验证

在 Qwen3-14B 的 PyPTO fused prefill 场景中，使用完全相同的权重、输入以及
全新的 KV cache，得到以下结果：

| 场景 | 结果 |
| --- | --- |
| 不执行 ATB warmup | logits 有限，`argmax = 17` |
| 执行 ATB warmup 后再执行 prefill | logits 全部为 `NaN` |
| ATB warmup 后执行 `torch.npu.synchronize()`，再执行 prefill | logits 仍全部为 `NaN` |
| 只在下游第一个 AIC kernel 入口加入 `SetAtomicNone()` | logits 恢复有限，`argmax = 17` |

该 A/B 修改没有改变任何模型计算逻辑，只是在第一个普通 store 前恢复了 AIC
的 atomic 状态。

## 建议修复

建议在 accumulation 工作完成后、所有可能的退出路径上恢复 non-atomic 状态。
例如，在 `pp_matmul_accum_atomic` 的 `switch` 之后增加：

```cpp
switch (masked_key) {
    // 现有分支。
}

AscendC::SetAtomicNone();
```

在使用新版 API 名称的工具链上，等价接口为
`AscendC::DisableDmaAtomic()`。

建议同时增加以下回归测试：

1. 使用 `enAccum = true` 执行一次 `LinearOperation`。
2. 随后执行一个独立 AIC kernel，向非零 GM buffer 执行普通 FP32
   L0C-to-GM 覆盖写入。
3. 检查所有参与 ATB 运算的 AIC 均保持覆盖语义。
4. 再增加一个显式同步后的测试分支，避免以后用“仅增加同步”的方式掩盖问题。

也建议检查 ATB 中其他调用 `SetAtomicAdd` 的 kernel 是否存在同类状态泄漏。

## 下游防御与责任边界

CANN 9.0 文档同时要求：静态张量编程生成的 kernel 应在入口调用
[`AscendC::InitSocState()`][init-soc-state]，因为之前的算子可能修改 AI Core 的
全局状态。PyPTO/PTO 也应补充这一防御性初始化，避免外部算子的遗留状态污染
自身 kernel。

不过，这一下游要求并不能消除 ATB 自身的边界问题：
`pp_matmul_accum_atomic` 明确打开了 atomic 模式，却没有关闭；而
[`SetAtomicAdd` 官方文档][set-atomic-add]也明确建议在使用后关闭该模式。在 ATB
内部恢复状态是最小且直接的修复，也能保护没有使用静态张量编程前端的其他
调用方。

## 当前临时规避方式

目前在 PyPTO Qwen3 prefill 路径前跳过
`_npu_matmul_add_fp32` warmup，可以避免数据损坏。但这样会失去原本的 ATB
预热效果，也不是通用修复。

[torch-npu-linear]: https://gitcode.com/Ascend/pytorch/blob/5462a1b3352e91893fdf65994b45a4b613b24861/third_party/op-plugin/op_plugin/ops/atb/LinearAtb.cpp
[atb-linear-runner]: https://gitcode.com/cann/ascend-transformer-boost/blob/578daaec1e5cf238ecfa8607a24ccbb606b46b6a/src/ops/ops_infer/linear/linear_ops_runner.cpp
[atb-matmul-operation]: https://gitcode.com/cann/ascend-transformer-boost/blob/578daaec1e5cf238ecfa8607a24ccbb606b46b6a/src/kernels/kernels/matmul/matmul_operation.cpp
[atb-atomic-kernel]: https://gitcode.com/cann/ascend-transformer-boost/blob/578daaec1e5cf238ecfa8607a24ccbb606b46b6a/src/kernels/kernels/matmul/pp_matmul_accum_kernel/op_kernel/pp_matmul_accum_atomic.cce
[set-atomic-add]: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0210.html
[init-soc-state]: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00094.html
[vllm-engine-init]: https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/engine/core.py#L249-L332
[vllm-executor-warmup]: https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/executor/abstract.py#L122-L126
[vllm-ascend-atb-warmup]: https://github.com/vllm-project/vllm-ascend/blob/341e8494a02d1a81b408a54ecd81202481163423/vllm_ascend/worker/worker.py#L721-L820
[vllm-ascend-workaround]: https://github.com/nalinaly/vllm-ascend/commit/b2fc1ecb615040bd8e74f1f0653389557040cfb8
