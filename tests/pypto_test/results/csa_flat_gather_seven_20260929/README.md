# 632dd00a 阶段出口：整块Gather后的同源码七档

七档为128K B4/B8/B16/B24、8K B16/B24/B32，长短权重8:2。
复用已完成的Sparse整块Gather候选128K/B16和8K/B24，仅补另外五档。
使用已验证且冻结的私有包 `.cache/csa-sparse-rope-flat-gather-369ad2c1-v1-candidate`，
variant为`pkg:dsv4_csa_sparse_rope_flat_gather_369ad2c1_v1`，等价生产632dd00a。
含Q与Sparse整块Gather，不含已否定的NZ、Q_B workers、O-A K512等候选。

Native复用最新标准七档：CANN9.2、npugraph_ex、dynamic=False、inplace/static/SuperKernel开启；
核内诊断仅关闭SuperKernel。PTO使用既有custom-op图，不因框架路径不同重测Native。
PyPTO88f605986/Simplera54c05095不变；mode2、atomic0、det0、EPLB关闭。
真实第二个CSA层权重与合成历史，5次预热20次正式事件，独立PyTorch profile及每档四窗DFX。

性能之后核对七档各自图重放的八类完整状态及保护区，跨版本逐元素证据为已完成的两代表档和H127/B3/padding。
其余五档不存在旧完整state，不能将自回放一致写成七档新旧逐元素一致；亦不代表新整模型token/DSpark通过。
本轮为阶段覆盖，不将跨轮Native/PTO或PTO旧/新表差额归因于单项优化；保留设备与采样任务来源。

完成后生成完整CSA/P95、Indexer/Sparse及其他任务核时明细，并汇集固定window_3七份原始泳道。
不挑最快窗口、不修改事件。沿用已有收集与命名方式；任务已完成退出0，七档各自八类图重放状态与28个DFX窗口覆盖通过。

[来源](source.json)、[补五档入口](run.sh)、[每档入口](run_side.sh)、[收集器](collect.py)、[泳道汇集](bundle.py)。

各上下文内batch等权，128K -11.869%、8K -1.528%，8:2 -9.801%。
七档均值均低于Native；8K/B24 P95仍高于Native，历史间歇长尾保持开放。
[正式结果](RESULTS.md)、[核内明细](TASKS.md)、[前置等待及上游参照](SCHEDULING.md)、
[七份泳道目录](download_pto_swimlanes/README.md)，压缩包为PTO_CSA_7cases_632dd00a_20260929.zip。
