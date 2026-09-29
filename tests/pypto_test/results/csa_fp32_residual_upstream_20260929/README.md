# 按pypto-lib组织FP32残差的HC_pre

参考pypto-lib2164563的hc_pre.py/hc_pre_gates：RMS与linear在同一inline函数中独立读取FP32残差，
RMS按8行、512列、stage4计算，allow_early_resolve=True。保留当前门控、Sinkhorn算术及尾行保护。
HC_pre混合后BF16舍入、attention BF16输出、HC_post FP32残差输入输出与上游相同。
已有O投影融合HC_post及四路残差复用继续使用；HC权重沿用当前默认缓存策略，不混入上游BYPASS调整。

首版FP32仅去副本，两档正式CSA为960.605→963.007μs、904.544→901.019μs，尚未证明收益。
本版在新完整私有包中实现上述上游组织，完成CPU编译，源文件只读后通过auto排队；
任务task_20260929_220910_164372022605自动分配设备2，首版为设备0。
每轮正式BF16/FP32内部同卡，两轮之间不混算绝对μs、不把差异全归因HC_pre。

两代表档各5预热20事件，性能后首调用八类状态、连续FP32输入的eager/graph及保护区检查。
长档FP32另采两个DFX窗，BF16 DFX复用首轮已完成的两个窗；明确属于跨轮/跨卡诊断资料。
不是完整FFN/HCA/Transformer或EP16模型验收，不把局部收益外推模型forward。

[来源与差异](source.json)、[相对首版补丁](candidate.patch)、[生成脚本](prepare.py)、
[完整CPU编译](compile_candidate.json)、[排队入口](run.sh)、[收集器](collect.py)、
[首版及基础FP32补丁](../csa_fp32_residual_20260929/README.md)、
[上游逐段对照](../../DSV4_FLASH_CSA_FP32_RESIDUAL_ESTIMATE.md)。

任务已完成退出0并释放设备；[正式结果](RESULTS.md)、[完整精度与原样本](evidence.json)。
