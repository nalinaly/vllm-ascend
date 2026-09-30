# O-A 两级流水：已接入HCA的NZ/N128分支

基线为生产0742f07c。此次保留八组、各组独立交给量化和O-B的依赖，
将O-A的L1读取改为K512、L0计算显式K128，两层stage2；不改外部cache或Native权重布局。
`integrated_v2`是接入版本：公共CSA默认PIPELINE_OA=False，HCA显式开启；ND和N256沿用原算术。

## 已有证据

真实第3层、CANN9.2、NZ2、BF16残差、atomic=0。每个性能行是独立同进程ABBA五组、每侧10次，μs。
不同候选不跨行比较绝对值，不把累计历史百分比相加。

| 候选/档位 | 控制 min/max/mean | 候选 min/max/mean | 组均值改善 |
| --- | --- | --- | --- |
| N128流水，128K/B16 | 549.00/608.75/568.650 | 552.00/570.25/558.650 | 5/5 |
| N128流水，8K/B16 | 439.75/515.00/454.675 | 442.25/460.75/448.500 | 3/5 |
| N256流水，128K/B16 | 590.75/620.75/604.500 | 587.25/606.00/597.750 | 4/5 |
| N256流水，8K/B24 | 596.50/644.00/612.725 | 598.75/625.25/612.875 | 3/5 |
| 首段剥离，对N128流水，128K/B16 | 551.75/615.50/570.875 | 563.75/581.25/573.400 | 1/5 |

长档N128首组对均值收益贡献较大；余四组均值仍改善约4.97μs，不宣称10μs是稳定收益下界。
完整输出与三类cache/state跨版逐bit、各自图重放和保护区通过。
8K/B16原报告在后续补位检查误报容量差异，故保留`failure.json/PADDING_FAILED`原状态。
校验器按Native无效slot值修复后，另一次只做功能的16→15→1→16补位重放通过，
不以该重试的两次计时替换上表性能。集成版128K/B4的4→3→1→4也通过完整逐bit与补位检查。
没有新增七档性能或整模型token验收。

独立同卡DFX的O-A结果：

| 实现 | worker数 | 核内min/max/mean | 总AIC占用/24 | O-A组跨度 |
| --- | ---: | --- | ---: | ---: |
| 原生产 | 64 | 22.42/29.12/25.773 | 68.729 | 83.68 |
| N128流水 | 64 | 14.06/24.62/20.489 | 54.638 | 70.64 |
| N256流水 | 32 | 27.44/46.64/38.560 | 51.413 | 81.00 |

N128核占用改善约20.5%。N256占用更低，但更长的单任务拖后量化/O-B交接，短档完整区间无改善，
保留独立候选，不设为默认。DFX中未修改的Attention也有波动，不能把所有变化归因O-A。

## Native来源与接口限制

最新本地ops-nn的`transpose_batch_mat_mul.cpp:71–88`，NZ走TransposeBatchMatMulKernel与MM_CFG_K_SHIFT。
原参考的`pp_matmul_ein_sum_kernel.h`是ND分支，只能说明双层流水思路，不能写成当前NZ实际模板。
NZ的`transpose_batch_mat_mul_base_tiling.cpp:302–350`按baseM/N/K分别求stepKa/Kb与双缓冲深度；
后续优先吸收该实际NZ策略。安装版Native的精确tiling仍不能由仓库源码单独证明。

当前N128实际Mat524288、Left65536、Right65536、Acc65536字节，L0两槽0/32768。
旧生产也已有双L0地址，不能再说“生产完全没有双缓冲”。收益来自读取粒度与循环/累加器组织。

显式窄行Acc种子需要`compact=True`及validshape；不设置会被AccCompactValid拒绝。
PyPTO把该参数标为内部接口，本次明确保留对当前工具链的依赖，不声称它是稳定公共API。
无该参数的`peeled`首段剥离CPU通过，但流水组退化提示且真机4/5组更慢，故不替换。
B4/B16窄行与补位已验证；以后工具链升级仍需关注此契约。

`integrated`条件调用独立inline函数因缺失tensor元数据编译失败，已改为SPMD内部constexpr分支。
`integrated_v2`完整依赖图、设备编译/load在NZ2与ND都通过。CPU通过不代替NPU验证。

## 文件

- [来源和冻结包](source.json)、[任务句柄](tasks.json)、[通过的检查与性能](summary.json)、[核内原始统计](incore.json)。
- [原始N128生成器](prepare.py)、[compact修正](prepare_compact.py)、[集成版生成器](prepare_integrated_v2.py)。
- [N256生成器](prepare_n256.py)、[N256容量](memory_n256.json)、[首段剥离容量](memory_peeled.json)。
- `results/hca_oa_pipeline_20260930/`保存原报告、失败报告、Pytorch profiling和DFX；未覆盖旧报告。

`kshift`只完成CPU编译、没有排真机。用户要求以Native为依据且只再做五点，
后续归入新的有限轮次审查；当前生产没有K次序改动。

CSA依赖调用补齐显式PIPELINE_OA=False：当前JIT依赖绑定不读取constexpr默认值。
完整CSA性能版依赖图、设备编译/load已通过，见compile_csa_dependency.json；不增加NPU复测。
