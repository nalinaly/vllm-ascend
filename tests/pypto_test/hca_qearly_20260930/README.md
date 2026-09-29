# Q投影提前解析的联动对照

基底`bef9f7fa`。仅把NZ Q投影的`allow_early_resolve`设为True，保留20个worker、
分块、真实依赖及核内流水；未叠加O-B候选。任务`task_20260930_020411_997802573`。

128K/B16，CANN9.2、NZ2、atomic0，同卡ABBA每侧18次设备span，单位μs：

| 实现 | min | max | mean |
| --- | ---: | ---: | ---: |
| base | 571.00 | 671.00 | 597.46 |
| qearly | 561.75 | 668.75 | 596.83 |

0.625μs差异不足以认定稳定收益。两轮base均值585.61/609.31，两轮候选582.50/611.17，
有明显进程漂移。最大值下降2.25μs也没有足够证据支持保留。暂不接入，不扩短档/七档测试。

独立DFX显示多task联动，不能只报Q投影组跨度缩短：

| 任务 | base开始–结束 | qearly开始–结束 |
| --- | --- | --- |
| Compressor投影 | 136.64–172.72 | 134.20–171.88 |
| Q投影 | 156.30–275.84 | 155.14–229.08 |
| Q反量化 | 282.44–330.68 | 237.98–324.14 |
| 压缩cache gather | 225.12–262.10 | 268.12–328.32 |
| attention AIV | 334.94–504.34 | 330.28–523.00 |

Q提前结束后，反量化与gather的任务组跨度都增加；attention虽然早4.66μs启动却晚18.66μs结束。
单窗口不能唯一归因为带宽或准入，但足以否定“Q变快就等于整层变快”的判断。
输出与全部cache/state逐bit一致，图重放和保护区通过；证据见
[result_h131072_b16.json](result_h131072_b16.json)。

`prepare.py`生成只读包和[qearly.patch](qearly.patch)，显式CPU编译通过。
`run.sh`经`task-submit`运行；汇总复用`hca_residual_reuse_20260930/collect.py`，
传`--experiment-root hca_qearly_20260930 --candidate-label qearly`。
