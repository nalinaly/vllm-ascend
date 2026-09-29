# 双query复用KV与完整UB累积：容量可行，当前流水未获益（2026-09-30）

本实验以第126节单query累计max候选为控制，它尚未接入生产。
生产算子保持0742f07c；不能将本实验对照误写成生产或Native。
CANN9.2/NZ2/atomic0/det0，正式第3层权重、BF16残差，128K/B16。
PTO以npugraph_ex、dynamic=False、inplace/static开启、SK关闭计完整mHC pre/norm/HCA/O/post。

## 为什么重新检查双query

第113节的全UB双query曾超过188416字节Vec限制，只能使用较慢的GM累积或半UB方案。
累计max先于BF16概率转换可减少PV合并临时值，因此重新核查容量，而非重跑原失败配置。
仍为24组MIX，压缩部分每次两个相邻query组成M128并共享KV；raw分别M64。
每个AIV负责一个完整64 heads query，在UB保留64×512 FP32累积。
四个Q组TaskId显式依赖、sink、mask、块顺序、cache和输出RoPE保持。

它与单query控制还有流水差异：两个KV L1槽、提前一tick QK、每对query结束后排空；
控制使用三个槽、跨query连续流水。不能只按KV读取减半估算完整收益。

## 先在CPU解决容量与生成码问题

| 版本 | Vec字节 | 状态 |
| --- | --- | --- |
| full_ub，四片N128累积 | 241152 | 超过188416，无NPU测试 |
| softmax改H16 | 236800 | 超容量，无NPU测试 |
| 同时输出按H8分组 | 222464 | 超容量，无NPU测试 |
| 累积更新拆八片N64，缩短softmax scratch生命周期 | 187648 | 容量内，但窄列assemble不被A3 tmov支持 |
| max/sum环改H16完整行 | 187648 | 编译/load通过，真机发现head slice错误 |
| H16静态展开 | — | SSA作用域失败，无NPU测试 |
| 显式extract动态H16行 | 187648 | 编译/load和真实层逐bit通过 |

最终Mat425984、Acc131072、Left/Right各65536字节。
没有改共享PyPTO/Simpler/PTOAS/PTO-ISA，仅在私有算子包调整活跃区间和支持的tile操作。
容量数据来自AllocateMemoryAddr中的地址上界，完整记录见[codegen_memory.json](codegen_memory.json)。

环布局版输出105094/1572864元素不一致、max_abs0.015625、RMSE0.000675566666；
三类cache/state及各自图重放精确。独立FP64 probe虽误差小，不能据此放过功能语义错误：
生成码将`tile.slice(softmax_m_iter, [16,1], [softmax_head,0])`地址固定为13312，
后几组head会读取第一组旧max，没有按预期实现累计max。

静态展开因SSA失败后，最终把[64,1]视为[4,16]，显式`tile.extract`动态整行，再reshape为[16,1]。
生成C++保留循环变量作为TEXTRACT行号，真实层输出恢复逐bit。
[slice_diagnosis.json](slice_diagnosis.json)保存前后生成码片段；[head_extract.patch](head_extract.patch)是修复增量。
`probe_result.json`的base是单query online，online是**已拒绝的环布局版**，不是最终修正版。
它仅证明小FP64误差不能代替实现正确性检查；没有倒设容差或宣称全模型token通过。

## 完整区间与核内结果

同进程5次预热、ABBA五组、每侧10个实际设备span。单位μs，min/max/mean。

| 档位 | 单query online控制 | 修正后的双query |
| --- | --- | --- |
| 128K/B16 | 546.750/598.750/557.400 | 559.500/591.000/570.400 |

mean增加13.000（2.332%），五个ABBA小组全部回退，P50增加17.25。
控制max在profile首样本，不能以max少7.75宣称稳定削峰。
完整x_out、swa、compressed、state四类状态跨版逐bit，各自eager/graph及保护区通过。
这是与单query online候选的等价检查，未覆盖online相对生产的算术变化验收。

独立同卡DFX，非上述计时窗口：

| 项目 | 单query online | 双query |
| --- | --- | --- |
| Attention AIC核内min/max/mean | 181.36/186.24/183.721 | 184.54/189.82/187.351 |
| Attention AIV核内min/max/mean | 184.48/189.48/187.075 | 193.74/203.78/199.297 |
| Attention AIV跨度 | 189.80 | 203.94 |
| Attention AIV启动/结束 | 302.60/492.40 | 310.98/514.92 |
| compressed gather跨度 | 57.00 | 41.56 |

尽管gather窗口更短，Attention更晚启动，核内也更慢；AIV mean多12.22μs。
更小Vector块、更多输出拼接以及两槽排空都可能抵消KV复用收益，当前证据不能独立量化各项。
不能把DFX核时变化与ABBA差值相加，也不能把双query失败归因于所有多query设计均不可行。
这版不接入、不扩七档/16卡；保留全UB容量解法和正确的动态head读取，后续有具体流水改进再复用。

## 证据与复现

[source.json](source.json)、[summary.json](summary.json)、[incore.json](incore.json)、
[rings_failure.json](rings_failure.json)、[tasks.json](tasks.json)、[task_status.json](task_status.json)。
中间失败由patch和容量表区分；源码冻结于source.json列出的只读私有包，禁止覆写。
patch使用零行上下文保存差异，应用时需要`git apply --unidiff-zero`，必须匹配所列父版本。
当前有效构建链：full_ub→softmax16_publish8→softmax16_publish8_n64_rings→head_extract。
各prepare脚本创建新包；CPU编译复用`hca_residual_reuse_20260930/compile.py`。

所有NPU测试经`task-submit --device auto --max-time 900`排队，复用
`hca_pair_screen_20260930/run.sh`和`hca_mix_schedule_20260930/run_dfx.sh`。
probe复用旧脚本，新增可选实验目录/两侧源码参数，不复制测试实现。
原始逐元素失败在`../results/hca_pair_online_20260930/long`；有效计时在`head_extract_long`。
泳道在`../results/hca_pair_online_20260930/dfx/{base,head_extract}/dfx/merged_swimlane.json`。
没有对性能回退方案补充整网测试。定向Ruff通过，全仓format.sh ci仍缺pre-commit。
