# RoPE提前派发：固定B24比较128K与8K

上一轮全局候选首轮8:2−0.467%、反序+0.338%，未采用；8K/B24两轮−0.684%/−1.143%。
长B16与短B24同时改变batch和长度，不能直接推断应该按长度分支。
因此补128K/B24，使token行数与已测短档相同（T=144）。不凭这一档推广全部长档或全部大batch。

复用csa_rope_early_revisit_20260929的两份完整冻结私有pkg，基底执行逻辑46cec3a0；
候选仍只有csa_rope_sign的allow_early_resolve=True，不改变真实依赖、算术或cache。
两份Python源均只读，原两入口解析/候选完整CPU编译证据继续有效，无源码修改不重复编译。

task_20260929_212522_12109358419正常auto排队，在设备1执行。CANN9.2、mode2、atomic0、det0，
真实第4层权重、合成历史、第二CSA层metadata复用；5预热20正式事件，另各4窗DFX。
性能后比输出、Top-K与六类cache/state的完整张量零容差，检查图重放和保护区。
Native只复用诊断已有对照，不新增Native性能基线或EP16测试。

正式事件、DFX窗口与此前各轮独立报告，不跨轮混加绝对μs，也不把局部交接改善代替完整CSA。
有收益再按实际工作量/长度设计同一算子内分支；无收益保留证据并转向更有效的关键链改动。

[来源](source.json)、[原编译证明](../csa_rope_early_revisit_20260929/compile_candidate.json)、
[原取舍](../csa_rope_early_revisit_20260929/decision.json)、[排队入口](run.sh)、[收集器](collect.py)。
