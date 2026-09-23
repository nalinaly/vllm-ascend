# A3 TDIV 高精度选项未生效：组件归因、Native 对比与最小复现

- 日期：2026-09-22。
- 范围：CANN9.0 / A3 上的 FP32 Tensor/Tensor `pl.div(..., high_precision=True)`。
- 最小复现不依赖 DSV4 模型、checkpoint、vLLM 或本仓库 Native custom op。
- 已执行任务：`task_20260922_125910_368689920621`，设备0，exit0，报告状态 `REPRODUCED`。
- 原始结果：[report.json](results/cann90_20260921/tdiv_high_precision_repro_v1/report.json)。
- 原始源码、IR/C++、输入输出及日志：[验证证据包](handoff/PTO_ISA_A3_TDIV_REPRO_20260922.tar.gz)（[SHA256](handoff/PTO_ISA_A3_TDIV_REPRO_20260922.tar.gz.sha256)）。
- 下文证据文件名默认位于包内 `results/cann90_20260921/tdiv_high_precision_repro_v1/`；原始生成文件按字节归档。

## 1. 结论与建议接收组件

**当前行为直接来自 PTO-ISA 的 A2/A3 TDIV 实现：高精度参数被接受，但没有选择不同的算法。**
PyPTO 生成的 PTO IR 保留了高精度属性；PTOAS 0.61 也生成了带 `HIGH_PRECISION` 的 TDIV 调用。
因此本例没有发现 PyPTO/PTOAS 丢失参数。

需要修正此前“高精度选项未生效”的表述：当前 PTO-ISA 的 TDIV 文档明确声明：

> Only available on A5, `PrecisionType` option is ignored on A3.

所以这里首先是**已声明的 A3 高精度 TDIV 能力缺口**，不能直接定性成违反 PTO-ISA 文档的实现 bug。
PyPTO 的 `high_precision` 参数说明没有同时展示这项平台限制，调用者容易误认为设置后已经生效。
完整文档快照：`isa_TDIV.md`。

| 组件 | 本次查证 | 建议处理范围 |
| --- | --- | --- |
| PTO-ISA | A2/A3 `TDIV_IMPL` 接受 `PrecisionType`，计算分支不使用它；最终调用 `vdiv` | A3 高精度向量除法能力的主要归属组件；评估补偿算法或明确的 fallback |
| PyPTO | `True` 已传入 IR；设备端标量除法可正常生成并运行 | 补充平台能力说明；在不支持的平台选择报错或明确提示，避免用户误判 |
| PTOAS | A3 / level3 的独立编译保留 `TDIV<...HIGH_PRECISION>` | 本例无参数传递修复需求；可与前端协商统一的不支持选项诊断 |

建议向 PTO-ISA 提交“**A3 FP32 TDIV 高精度能力请求**”，并向 PyPTO/PTOAS 同步“**不支持选项的能力提示**”。
本次归因与复现没有修改上述组件的实现。

## 2. 同一 A3 上 Native 为什么能满足这里的精度要求？

**Native QR 使用设备端标量除法；PTO TDIV 使用向量除法。两者并非同一条计算路径。**

Native 实现在本仓库
`csrc/attention/rms_norm_dynamic_quant/op_kernel/rms_norm_dynamic_quant_normal_kernel.h`：

```cpp
// 归约后，每行计算一个归一化系数。
float rstdLocalTemp = 1 / sqrt(squareSumTemp * this->aveNum + this->eps);

// 每行计算量化乘数及反量化尺度。
maxTemp = tmpTensor[rid * this->numLastDimAligned].GetValue(0);
scaleTemp = this->quantMaxVal / maxTemp;
scaleTensor.SetValue(rid, 1 / scaleTemp);

// 用行系数执行向量乘法。
Muls(srcSlice, srcSlice, scaleTemp, this->numLastDim);
```

这里的 `GetValue/SetValue` 在 `__aicore__` 内操作设备侧数据，并有 V→S、S→V 事件依赖；
不是 Python `.item()` 或把值送回 CPU 计算。Native 每行只需要几个标量除法，行内主体仍为向量运算。
源文件与当前安装 vendor 的对应文件已核对哈希，见
[source_chain.json](results/cann90_20260921/tdiv_high_precision_repro_v1/source_chain.json)。
相关源码摘录：`native_qr_rstd_excerpt.txt`、
`native_qr_scale_excerpt.txt`。

| 路径 | 同一 A3 上的实际选择 | 本次观察 |
| --- | --- | --- |
| Native QR | 行归约后，用设备端标量 C++ 表达式计算除法，再进行向量乘法 | DSV4 Native 对照采用此路径 |
| PTO `pl.div(Tensor, Tensor)` | `TDIV` → A2/A3 `DivOp` → `vdiv` | 存在相对正确舍入 FP32 结果的 1 ULP 差异 |
| PTO `pl.div(..., high_precision=True)` | `TDIV<HIGH_PRECISION>` → 同一个 A2/A3 `DivOp` → 同一个 `vdiv` | 与默认模式逐 bit 相同 |
| PTO 设备端标量 `left / right` | `arith.divf` → 设备 C++ 的 `float / float` | 4096 个复现输入全部与正确舍入参考一致 |
| PTO-ISA A5 高精度 TDIV | 源码显式选择 FP32 `DivDiffCompensationFloatImpl` / FP16 `DivIEEE754HalfImpl` | 本次只核对源码，没有 A5 真机精度结论 |

因此不能解释成“A3 硬件完全做不到这种精度”：**同一块 A3、同一批输入，设备端标量路径已经做到。**
这也不能证明 A3 的单条向量 `vdiv` 具有可打开的高精度模式；标量表达式和向量 intrinsic 的结果不同，
并不要求它们有相同的执行方式、吞吐或舍入行为。
本次证据到生成的设备 C++ 和真机结果为止，未反汇编两条路径，因此不指定标量表达式最终对应的机器指令，
也不据此声称其硬件实现或延迟与 `vdiv` 相同。

对于“PTO-ISA 是否故意不做”，可以确认的是：**A3 忽略参数是源码和文档明确列出的支持范围选择，
并非参数偶然丢失。** 现有材料没有解释原因，不能据此认定是硬件绝对限制、性能取舍或开发排期。
具体历史原因和 A3 向量高精度实现方案需要维护者确认。A5 的分支是架构相关实现，不能仅删除 A3 限制声明
或把参数改名就认为补齐了能力。

## 3. 已核对的版本

| 项目 | 本次实际版本 |
| --- | --- |
| 硬件 | `Ascend910_9392`，A3，最小复现使用逻辑设备0 |
| CANN | `/usr/local/Ascend/cann-9.0.0` |
| PyTorch / torch_npu | `2.10.0+cpu` / `2.10.0.post2` |
| PyPTO | `02c0026993b08353e6cee4fdb93e86eddabb8701`，`feat/kernel-mode-integration-test` |
| Simpler | `e914837d540a899dfce0cf48f2adac91e3884930`，同名调试分支 |
| PTOAS | `0.61`，`.cache/csa/ptoas-0.61/bin/ptoas` |
| PTO-ISA | `03e45c4bda48a6909feb239f0acc45f04176c2a1`，`simpler/build/pto-isa` |
| Runtime | `platform=a2a3`，`runtime=tensormap_and_ringbuffer` |

这是上述安装组合的结论，不代表其他版本已验证。A2 与 FP16 未做真机复现；A2/A3 共享实现文件
只提供源码证据，不能替代其他平台和 dtype 的数值测试。

## 4. 参数在哪一层被忽略？

### 4.1 PyPTO：属性保留

`python/pypto/language/op/tensor_ops.py::div` 将 `high_precision` 交给 IR。
`src/backend/common/pto_ops_elementwise.cpp::MakePrecisionCodegenPTO` 输出精度属性。
本次实际生成的 `vector_input.pto` 同时含有：

```text
pto.tdiv ins(...) outs(...)
pto.tdiv ins(...) outs(...) {precisionType = #pto<div_precision high_precision>}
```

测试使用两个相同 shape 的 FP32 Tensor，没有 scalar rhs、广播或 Tensor/Scalar canonicalization 干扰。

### 4.2 PTOAS：模板参数保留

实际生成的 `vector_generated.cpp`：

```cpp
TDIV(v64, v47, v56);
TDIV<pto::DivAlgorithm::HIGH_PRECISION>(v72, v47, v56);
```

另外直接运行 PTOAS 编译已保存的 IR，同样保留该参数；不需要再次运行 PyPTO 或启动 NPU。
独立结果：[standalone_ptoas.json](results/cann90_20260921/tdiv_high_precision_repro_v1/standalone_ptoas.json)。

### 4.3 PTO-ISA：公共入口传递，A3 后端忽略

公共入口 `include/pto/common/pto_instr.hpp` 调用 `TDIV_IMPL<PrecisionType>(...)`。
但 `include/pto/npu/a2a3/TDiv.hpp` 中 `TDIV_IMPL` 没有根据 `PrecisionType` 分支，后续调用链
也没有携带这个精度参数；`DivOp::BinInstr` 直接调用 `vdiv(...)`。

完整安装文件快照：`isa_TDiv.hpp`。
供比较的 A5 分支摘录：`isa_a5_excerpt.txt`。

## 5. 不依赖模型的最小真机复现

文件：[repro_tdiv_high_precision.py](repro_tdiv_high_precision.py)，
本机队列包装：[run_tdiv_high_precision_repro.sh](run_tdiv_high_precision_repro.sh)。

复现只做三种除法：

```python
# 设备 kernel 内：相同输入，两个输出。
normal[row:row + 1, :] = pl.div(left, right)
precise[row:row + 1, :] = pl.div(left, right, high_precision=True)

# 独立设备 kernel 内：逐元素标量对照。
left = pl.read(numerator, [row, column])
right = pl.read(denominator, [row, column])
pl.write(values, [0, column], left / right)
```

输入 shape 为 `[16,256]`，共4096个值，CPU随机种子为20260922。分子包括1、127和一般 FP32；
分母非零，输入与商均为正、有限、normal FP32，排除了除零、NaN、溢出和 subnormal 干扰。
CPU先用 FP64 除法后转 FP32 产生参考候选，再逐元素用精确有理数检查相邻 FP32 的最近偶数舍入；
所有4096个参考值都经过该检查。CPU仅用于输入生成及结果断言。

在本机仓库根目录执行，使用一个新的结果目录：

```bash
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto
task-submit --device auto --max-time 600 \
  "bash $PWD/tests/pypto_test/run_tdiv_high_precision_repro.sh $PWD/tests/pypto_test/results/tdiv_repro_new"

task-submit --status <返回的任务ID>
task-submit --log <返回的任务ID>
```

维护者在其他环境准备相应 PyPTO/PTOAS/PTO-ISA、Torch/torch_npu 和 runtime 后，可用自己的硬件队列
调用下面的独立脚本；不需要复制 DSV4 仓库其余计算代码：

```bash
python repro_tdiv_high_precision.py \
  --device <队列分配的设备号> --output-dir <新目录> --isa-root <实际PTO-ISA目录>
```

`--isa-root` 用于保存版本与源码证据，不用于切换实际编译器 include；应与运行环境真正使用的 ISA 一致。
程序退出0表示复现流程完成，**`REPRODUCED` 表示观察到了所报告的行为，不是高精度验收通过。**

## 6. 本次实测结果

| 比较 | 不同元素数 / 4096 | 最大 ULP |
| --- | ---: | ---: |
| 高精度选项 vs 默认 TDIV | 0 | 0 |
| 默认 TDIV vs 正确舍入 FP32 | 244 | 1 |
| 高精度选项 vs 正确舍入 FP32 | 244 | 1 |
| 设备标量除法 vs 正确舍入 FP32 | 0 | 0 |

一个可以直接固定为回归输入的例子：

| 数据 | 十进制 | FP32 位型 |
| --- | ---: | --- |
| 分子 | 1.0 | `0x3f800000` |
| 分母 | 0.00003821843711193651 | `0x38204cbc` |
| 默认 / 高精度选项输出 | 26165.3828125 | `0x46cc6ac4` |
| 标量 / 正确舍入参考 | 26165.380859375 | `0x46cc6ac3` |

全部输入与输出：`tensors.pt`。
该结果证明这组输入上默认 TDIV 与标量路径的差异，不推导任意输入的最大误差界，也不说明默认 `vdiv`
违反其自身精度规范。本次没有做性能对照，逐元素标量复现也不作为高吞吐向量实现建议。

## 7. 只核对 PTOAS 的复现

在仓库根目录解开证据包，再使用保存的 `vector_input.pto`，不启动 NPU：

```bash
tar -xzf tests/pypto_test/handoff/PTO_ISA_A3_TDIV_REPRO_20260922.tar.gz \
  --strip-components=1 -C tests/pypto_test
```

```bash
source ../env.sh
ptoas tests/pypto_test/results/cann90_20260921/tdiv_high_precision_repro_v1/vector_input.pto \
  --pto-arch=a3 --pto-level=level3 --enable-insert-sync -o /tmp/tdiv_a3.cpp
rg 'TDIV' /tmp/tdiv_a3.cpp
```

应看到默认调用和 `TDIV<pto::DivAlgorithm::HIGH_PRECISION>`。这里的 IR 已有显式 tile 地址，因此使用
`level3`；用 `level2` 会报 `unexpected 'addr' operand`，与高精度问题无关。
这条命令只证明参数传递，不替代第5节设备精度实验。

## 8. DSV4 影响与已有处理

DSV4 QR 每行依次计算 `rstd`、`127/amax`、`1/quant_multiplier`。很小的 FP32 差异可能使量化值
跨过半整数舍入边界，导致 INT8 相差1，并进一步影响投影、Indexer 和 Top-K。

在原 B40 参考用例中，仅把这三个逐行除法改成设备标量计算后，QR INT8 差异由8个降为0个，
最终输出超出冻结容差的元素数由3832降为351。其余误差另有来源，不能归到 TDIV 这一项：

| 原日志中的现象 | 归属/处理 |
| --- | --- |
| Indexer QB 两个 scale 的相乘顺序 | 浮点结合顺序与 Native 不同；本仓库调整为先合并尺度，属于接入计算边界对齐 |
| QR 剩余 scale 末位差 | 后续定位到平方根舍入；独立于高精度 TDIV 选项，不能用本复现证明 sqrt 的组件责任 |
| QA BF16 累加差异 | 后续定位到 Native MatMul 的 K 块遍历顺序；独立于 TDIV |
| 合法 `None` slot 的 `data_ptr` 失败 | 测试地址签名问题，已在本仓库修正 |
| 临时 cast Tensor 的 metadata 推断失败 | 独立的 lowering 问题，已有参数传递 workaround；若单独报编译器问题，需要另外制作最小复现 |

证据包内的 `DSV4_QR_CONTEXT.md` 保留验证日志第43～46节的原始记录。当前 workaround 仅在
`vllm_ascend/ops/pypto/deepseek_v4_flash_dspark/qkv_proj_rope.py` 内，未扩展到依赖仓库。

## 9. 建议组件维护者确认和验证的事项

1. PTO-ISA 明确是否计划提供 A3 FP32 高精度 TDIV；若提供，定义实际误差指标和边界语义。
2. 用本报告固定输入验证高精度路径确实选择独立实现；“模板里出现 HIGH_PRECISION”不作为数值通过依据。
3. 若继续保持 A3 不支持，PyPTO/PTOAS 明确提示或拒绝该请求，并同步 API 文档的支持平台。
4. 补充 FP16、负数、极值、subnormal、零及非有限值的组件测试；这些不在本次4096个 normal FP32 样本范围内。
5. 实现变更后运行本最小复现，再重跑 DSV4 QR 的相同 QA 输入对照和完整 CSA；性能另行计量。

本报告没有要求修改 Native baseline、放宽 CSA INT8 一致性规则，或把整个向量计算改成标量。
