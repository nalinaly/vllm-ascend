# HCA Q投影与反量化的联动调度

基底为c6792787加已验证的residual复用（生产提交2cb8714b仅额外清理了无用import）。
先检查单卡128K/B16；CANN9.2、NZ2、atomic0、det0、BF16残差、正式第3层权重。
本轮不修改算术、任务分块、task依赖、缓存布局或共享运行时。

此前独立泳道中，同样的Q投影20个worker，首开到末结束跨度可从71.92到126.30 μs；
48个worker的反量化任务跨度48.52到83.52 μs。核内耗时和分批启动同时影响收尾，
不能只按20/24或48/48的静态核数判断是否饱和，也不能把单个task的变化当整层收益。

本轮三方同卡正反对照：

| 变体 | Q投影Cube | Q反量化/RMS/RoPE AIV |
| --- | --- | --- |
| base | 原配置20 worker | 原配置48 worker、early resolve |
| syncq | 同样20 worker，增加sync_start | 不变 |
| syncboth | 同syncq | 同样48 worker、early resolve，增加sync_start |

Q投影会与KV和Compressor投影重叠；反量化又可能与gather/state写回重叠。
同步启动可能缩短任务首尾跨度，也可能等核或减少有效重叠，故单改与联动同时比较。
继承原有的真实依赖，特别是Compressor pool→state_commit的WAR顺序，不以并发为由删除。

`prepare.py`完整冻结三包并设Python源为只读；两候选变化记录在patch中。
复用上一轮`compile.py`显式解析依赖图、CPU编译并load，三份均通过。
`run.sh`按base/syncq/syncboth/syncboth/syncq/base执行，各20次事件计时、9次profiler重放，
然后分别采一个独立DFX窗口。`collect.py`保留全部单次min/max/mean、进程中位数、
核内耗时、组启动跨度和前后链路重叠；一次性比对保存的完整输出/cache/state。

任务task_20260930_012821_4472725985，结果根目录：
`../results/hca_qchain_schedule_20260930/h131072_b16/`。
任务已完成exit=0。每侧18次单次profiler重放，单位μs：

| 变体 | min | max | mean | mean相对base |
| --- | ---: | ---: | ---: | ---: |
| base | 562.00 | 645.00 | 582.60 | — |
| syncq | 568.25 | 643.25 | 590.25 | +7.65 |
| syncboth | 588.75 | 649.00 | 607.90 | +25.31 |

完整输出及cache/state与首份base逐bit一致，图重放和保护区通过。
syncq的max低1.75 μs，同时min/mean升高，小样本不能据此宣称有稳定长尾收益；
syncboth三项均变差。本轮两个候选都不接入，不追加短档重复确认负收益。

独立DFX的联动证据（每侧一个窗口，不能冒充18次的分布）：

| 时间段 | base | syncq | syncboth |
| --- | --- | --- | --- |
| Q首次启动→最后结束 | 151.70→280.40 | 171.00→241.80 | 175.16→244.76 |
| Q启动跨度 | 63.78 | 0.62 | 2.16 |
| 压缩KV gather首开→末结束 | 216.26→255.68 | 246.88→315.58 | 240.32→284.68 |
| 反量化首开→末结束 | 286.72→336.08 | 256.58→318.68 | 290.46→343.70 |
| Q最后结束→反量化首次启动 | 6.32 | 14.78 | 45.70 |
| attention首次启动 | 338.96 | 320.20 | 349.12 |

Q的首尾跨度确实被同步启动缩短；但生产者提前结束不保证消费者更快。
syncq中的反量化与gather重叠；syncboth中的反量化在gather末结束后5.78 μs才起跑，
观察到两者串行化，等待转移至下一段。不能把Q的约58 μs跨度减少计作整层收益。
syncq这一张DFX窗口的attention更早、总worker跨度也更小，但独立ABBA的mean更慢，
再次说明单张泳道是机制线索，不能替代整层计时验收。
详细样本和原始文件路径见`result_h131072_b16.json`。

下一轮incore排查以最新ops-transformer为参考；已经核对
`experimental/attention/sparse_attn_sharedkv/op_kernel/arch22/sparse_attn_sharedkv_scfa_block_vector.h`：
Native的`RowDivs`也调用Div，因此不能宣称“Native输出归一化已用倒数乘法，而PTO没有”。
更有针对性的待查项是完整压缩块的mask处理：Native按实际有效列数进入softmax，
PTO每块还执行mask加载、bias、fillpad和乘mask。任何快路径必须同时证明长度及所有页有效，
否则保持现有通用分支；不能只根据128K或页表容量跳过保护。
