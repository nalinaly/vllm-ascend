# [Bug] PyPTO 生成的静态 Tensor kernel 入口未初始化 AI Core 全局状态

## 问题概述

PyPTO 的 PTO backend 在生成 `kernel_entry` 时，直接解析运行时参数并调用
PTOAS 生成的函数，没有在入口调用 `AscendC::InitSocState()` 或等价初始化。

CANN 对静态 Tensor 编程有明确约束：前序算子可能修改 AI Core 的全局配置
状态，因此静态 Tensor kernel 必须在入口进行 SoC 状态初始化。当前缺失会使
PyPTO kernel 的语义依赖同一物理 AI Core 上之前运行过什么算子。

本问题已经在真实 NPU 上复现。一个前序 ATB accumulation kernel 遗留
atomic-add 状态后，PyPTO 生成的普通 FP32 L0C-to-GM `TSTORE` 不再覆盖输出，
而是错误地对原值做累加：预期写入 `64`，实际写入 `100 + 64 = 164`。在
Qwen3-14B fused prefill 中，该状态污染会使 logits 全部变为 `NaN`。

ATB 未清理自己打开的状态是一个独立的上游 bug；但 PyPTO 作为静态 Tensor
kernel 生成器，也可以并且应当在自身入口完成防御性初始化。两个修复并不
冲突，责任边界见本文后面说明。

## 影响范围

- 至少 PyPTO 生成的 AIC kernel 已在 A2/A3 真机上确认受影响。
- `_generate_kernel_wrapper()` 是通用入口生成逻辑，AIC/AIV 以及 split kernel
  都应检查和统一修复。
- 问题不限于 ATB。任何没有正确清理 AI Core 全局状态的前序算子，都可能污染
  后续 PyPTO kernel。
- atomic-add 只是目前已经观察到的一种状态。CANN 的初始化还涉及 mask、
  padding、L1 3D 等架构相关状态，单独修补某个 `TSTORE` 不足以覆盖完整契约。

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
| torch_npu | `2.12.0+git5462a1b` |
| PyPTO | `f3b6e7f0c916435f710bc195a6113168d701cf80` |
| PTO-ISA | `83d01313d9bfc247c4b7c8bcf969d1019f0d106f` |

## CANN 的静态 Tensor kernel 入口约束

CANN 9.0 的 [`AscendC::InitSocState()` 文档][init-soc-state]说明，AI Core 的
全局状态可能被前序算子修改；静态 Tensor 编程需要在 kernel 入口主动调用该
接口，否则可能出现精度错误，甚至执行异常。使用 `TPipe` 的编程框架会自动
初始化，而 PyPTO 当前生成的是不创建 `TPipe` 的静态 Tensor kernel，因此不
具备这层自动保护。

在本机 CANN 9.1.0 的 A2/A3 实现中，`InitSocState()` 首先执行
`set_atomic_none()`，并继续恢复 mask 以及 AIC/AIV 相关的默认状态。这说明：

1. atomic 模式确实是可跨 kernel 遗留的 AI Core 配置状态；
2. 入口初始化应使用完整的架构 API，而不应只针对本次暴露的 atomic 症状打补丁。

这里的 atomic-add 不是“只对某个地址执行一次的原子指令”。地址仍由后面的
store 指令提供，而 AI Core 上的 atomic 配置决定该 store 是覆盖 GM，还是对
GM 做原子 read-modify-write。任务完成或 stream 同步不会自动恢复这个配置。

## PyPTO 的具体代码 bug

### 1. 生成的 `kernel_entry` 没有任何状态初始化

[`python/pypto/backend/pto_backend.py`][pypto-wrapper] 中的
`_generate_kernel_wrapper()` 当前生成如下。截至 PyPTO 上游 `main` 的
`47f98653cf3ced2749b39126ac9db10d3eaf01fb`，这里仍没有入口初始化：

```python
wrapper_func = (
    "// --- Kernel entry point ---\n"
    'extern "C" __aicore__ __attribute__((always_inline)) '
    "void kernel_entry(__gm__ int64_t* args)\n"
    "{\n"
    f"{runtime_subblock_setup}"
    f"{spmd_args_setup}"
    f"{subblock_arg_setup}"
    f"{unpacking_code}\n"
    f"{sdma_setup}"
    f"    // Forward to ptoas-generated function\n"
    f"    {func.name}({call_args});\n"
    "}\n"
)
```

也就是说，入口后的第一批操作已经开始读取 runtime identity、解析参数并进入
PTOAS 函数，但整个入口没有 `InitSocState()`。

从本次复现实际生成的 AIC 源码中也可以直接看到：

```cpp
// ptoas-generated body
wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
TSTORE(v28, v23);

// PyPTO wrapper
extern "C" __aicore__ __attribute__((always_inline))
void kernel_entry(__gm__ int64_t* args)
{
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    // 参数解析省略
    overwrite_with_matmul_incore_0(/* ... */);
}
```

### 2. 普通 `TSTORE` 不会清除继承来的 atomic 状态

PTO-ISA 的
[`TSTORE_IMPL`][pto-tstore] 默认模板参数是
`AtomicType::AtomicNone`，但这里的 `AtomicNone` 只表示“本次 store 不主动打开
atomic”，不表示“先将硬件状态重置为 non-atomic”。相关逻辑为：

```cpp
if constexpr (currentAtomicType == AtomicType::AtomicAdd) {
    SetAtomicAdd<typename GlobalData::DType>();
}

// 执行 TStore / TStoreAcc / TStoreMat

if constexpr (currentAtomicType == AtomicType::AtomicAdd) {
    SetAtomicNone();
}
```

显式 atomic store 会成对执行 `SetAtomicAdd` 和 `SetAtomicNone`；普通 store
则两者都不执行。因此，只要 kernel 入口已经处于 atomic-add 模式，默认
`TSTORE` 就会继承该模式。

这不是要求 PTO-ISA 在每条普通 store 前重复重置状态。更合适的边界是：PyPTO
在每次生成的静态 Tensor kernel 入口建立一个确定的初始状态，之后 PTO 指令
只需维护本 kernel 内部的状态切换。

## vLLM 场景中的完整触发顺序

真实模型中并不是 PyPTO 主动调用 ATB，而是 vllm_ascend 在接收第一个请求前
执行 ATB warmup：

```text
vLLM EngineCore._initialize_kv_caches()
  -> model_executor.compile_or_warm_up_model()
    -> collective_rpc("compile_or_warm_up_model")
      -> vllm_ascend NPUWorker.compile_or_warm_up_model()
        -> NPUWorker._warm_up_atb()
          -> torch_npu._npu_matmul_add_fp32(...)
            -> ATB pp_matmul_accum_atomic
              -> SetAtomicAdd<float>()，返回前未清理

随后第一个真实请求：
  -> PyPTO fused prefill kernel_entry
    -> 未调用 InitSocState()
      -> 普通 AIC TSTORE 继承 atomic-add 模式
```

vllm_ascend 上游当前的
[`NPUWorker._warm_up_atb()`][vllm-ascend-warmup] 实现为：

```python
def _warm_up_atb(self):
    x = torch.rand((2, 4), dtype=torch.float16).npu()
    weight = torch.rand((2, 4), dtype=torch.float16).npu()
    c = torch.rand((4, 4), dtype=torch.float32).npu()
    torch_npu._npu_matmul_add_fp32(x, weight, c)
```

该调用的结果被丢弃，目的只是预热 ATB。Python 调用栈结束后，物理 AIC 上的
atomic 配置仍可能保留，因此随后调度的 PyPTO kernel 会受到影响。

## 最小复现

下面的脚本包含三个步骤：

1. 先覆盖所有 AIC，并建立普通 store 的正确基线；
2. 调用一次 ATB `_npu_matmul_add_fp32` 污染其中一个 AIC；
3. 再执行同一个 PyPTO 普通 store，观察 `64` 变成 `164`。

测试机有 20 个 cube core；其他设备需要相应修改 `AIC_COUNT`。

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
    # 显式 atomic store 会先 SetAtomicAdd，并在 store 后 SetAtomicNone。
    # 每个 AIC 启动一个 block，用它清除之前测试可能遗留的状态。
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
        # 默认 store 的语义必须是覆盖写入。
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
    out = torch.full((AIC_COUNT * 64, 64), 100.0, dtype=torch.float32)
    ordinary_store(a, b, out, config=config)
    return out


def run_atb_accum():
    x = torch.ones((2, 4), dtype=torch.float16, device="npu:0")
    weight = torch.ones((2, 4), dtype=torch.float16, device="npu:0")
    accum = torch.zeros((4, 4), dtype=torch.float32, device="npu:0")
    torch_npu._npu_matmul_add_fp32(x, weight, accum)
    # 特意同步，证明 synchronize 不会重置 AI Core 状态。
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
ASCEND_RT_VISIBLE_DEVICES=1 python repro_pypto_missing_init_soc_state.py
```

某次真机输出：

```text
ATB 结果的唯一值： [2.0]
基线结果的唯一值： [64.0]
ATB 后结果的唯一值： [64.0, 164.0]
每个 block 的首元素： [..., 64.0, 164.0]
```

具体受影响的 block 取决于 AIC 调度。关键断言不是固定 block 下标，而是 ATB
之后至少一个普通 store 产生了 `164`。

## 为什么该结果能定位到入口初始化缺失

两个输入都是全 `1` 的 `64 x 64` 矩阵，因此 matmul 的每个输出严格为 `64`；
输出 tensor 在 kernel 启动前被初始化为 `100`。

- 正确的普通 store：`out = 64`。
- 继承 atomic-add 后：`out += 64`，所以得到 `164`。

ATB 自身的输出全部为 `2`，说明不是 matmul 数值精度错误。加入
`torch.npu.synchronize()` 后问题仍存在，排除了异步执行顺序。只在下游第一个
AIC kernel 入口加入状态清理后，`164` 消失且模型 logits 恢复正常，因果链条
与 `InitSocState()` 的官方用途一致。

## 真实模型 A/B 结果

在完全相同的 Qwen3-14B 权重、输入和全新 KV cache 下：

| 场景 | 结果 |
| --- | --- |
| 不执行 vllm_ascend ATB warmup | logits 有限，`argmax = 17` |
| ATB warmup 后执行 PyPTO prefill | logits 全部为 `NaN` |
| ATB warmup 后先 `torch.npu.synchronize()` | logits 仍全部为 `NaN` |
| 只在首个下游 AIC 入口清除 atomic 状态 | logits 有限，`argmax = 17` |

## 建议修复

### 1. 在每个生成的硬件 `kernel_entry` 最前面初始化

建议由 `_generate_kernel_wrapper()` 统一发出 SoC 状态初始化，并使它成为
`{` 后的第一项设备状态操作，早于 runtime subblock/SPMD setup、参数解析以及
PTOAS 函数调用。语义示意如下：

```cpp
extern "C" __aicore__ __attribute__((always_inline))
void kernel_entry(__gm__ int64_t* args)
{
#if !defined(__CPU_SIM)
    AscendC::InitSocState();
#endif

    // 现有 runtime identity、参数解析和 PTOAS 调用。
}
```

实现时需要在硬件编译路径引入对应的 CANN 公共 header 和 include directories。
如果不希望 backend 直接依赖 AscendC header，也可以在 PTO-ISA 中提供一个跨
架构的入口初始化封装：A2/A3、A5 分别调用平台 API，CPU simulator 实现为空
操作。重点是使用完整的 SoC 初始化语义，而不是只发出一条
`set_atomic_none()`。

### 2. 不建议仅修改普通 `TSTORE`

在每条普通 `TSTORE` 前调用 `SetAtomicNone()` 可以遮蔽当前症状，但有三个
问题：

1. 无法恢复其他可能污染的 AI Core 全局状态；
2. 在一个 kernel 内重复增加不必要的状态写入；
3. 将 kernel 边界契约分散到了具体指令实现中。

入口调用 `InitSocState()` 与 CANN 静态 Tensor 编程模型一致，边界也更清晰。

## 建议回归测试

1. codegen 单测：检查所有生成的 `kernel_entry` 在进入现有 setup 逻辑前包含
   一次初始化。
2. AIC 真机测试：先污染 atomic 状态，再运行本文的普通 L0C-to-GM store，
   所有 block 都必须得到 `64`。
3. AIV 真机测试：构造 vector store 的同类状态污染用例，确认通用 wrapper
   同样生效。
4. simulator 测试：`a2a3sim`/`a5sim` 仍可编译运行，初始化封装在模拟器中为
   明确的 no-op。
5. 架构编译测试：至少覆盖 A2/A3 和 A5，避免硬编码单一架构接口。
6. 显式 atomic store 测试：入口初始化不能破坏 PyPTO 自身合法的
   `AtomicType::Add` 语义。
7. Qwen3-14B 端到端测试：恢复 vllm_ascend ATB warmup 后，fused prefill
   logits 仍应为有限值。

## 与 ATB bug 的责任边界

本次触发链同时暴露了两个可以独立修复的问题：

- ATB：`pp_matmul_accum_atomic` 打开 `SetAtomicAdd<float>()` 后没有在返回前
  关闭。ATB 应修复该算子，以保护所有下游调用方。
- PyPTO：生成静态 Tensor kernel 时没有按 CANN 约束初始化入口状态。PyPTO
  应补齐防御性初始化，以防任何前序算子的遗留状态污染自身。

即使 ATB 修复，PyPTO 的入口初始化缺失仍然存在；即使 PyPTO 修复，ATB 也
不应继续向其他消费者泄漏状态。因此建议分别提交和跟踪两个 issue。

## 当前临时规避

- vllm_ascend 已可针对 PyPTO Qwen3 路径跳过 ATB warmup，参考
  [`b2fc1ecb`][vllm-ascend-workaround]。
- 在首个 PyPTO AIC 入口执行 `SetAtomicNone()` 能规避当前 atomic 症状，但不
  等价于完整的 `InitSocState()`。
- `torch.npu.synchronize()` 不是有效规避方式。

[pypto-wrapper]: https://github.com/hw-native-sys/pypto/blob/47f98653cf3ced2749b39126ac9db10d3eaf01fb/python/pypto/backend/pto_backend.py#L801-L816
[pto-tstore]: https://github.com/hw-native-sys/pto-isa/blob/83d01313d9bfc247c4b7c8bcf969d1019f0d106f/include/pto/npu/a2a3/TStore.hpp#L296-L339
[init-soc-state]: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00094.html
[vllm-ascend-warmup]: https://github.com/vllm-project/vllm-ascend/blob/341e8494a02d1a81b408a54ecd81202481163423/vllm_ascend/worker/worker.py#L790-L820
[vllm-ascend-workaround]: https://github.com/nalinaly/vllm-ascend/commit/b2fc1ecb615040bd8e74f1f0653389557040cfb8
