# 512列实验的PV双缓冲候选：CPU完成，尚未提交NPU

父版本是未采用的macro512_qkpipe，不是生产128列Attention。
候选在保留四个输出累加器及原BF16概率策略的前提下，尝试PV的K维预取。

- pvpipe首版：四个N列的Right提取被一起展开，Right需求262144字节超过65536，编译失败。
- pvpipe_nested：列循环每次仅消费一个Right，但elif中的tuple yield触发SSA限制。
- pvpipe_nested_v2：四个显式if/else各更新一个累加器，CPU依赖图、设备编译/load通过。

最终版有两个131072字节的PV L1槽，Right使用0/32768两个地址，
四个Acc各32768字节，容量可容纳。CPU通过不证明设备数值或性能通过。
原raw零次迭代、奇数压缩微块尾部等仍需设备验证，未将它们记作已验收。

用户已将主线改为按GAP从大到小优化；LOG135显示O-A显著落后、Attention本体已接近Native。
因此该候选只保留编译证据，尚未提交NPU任务；当前优先O-A，不增加小效应扩测。
没有生产采用、七档或模型验收结论。

来源与逐版失败见[source.json](source.json)，
最终编译记录见[compile_pvpipe_nested_v2.json](compile_pvpipe_nested_v2.json)。
依次运行三个prepare脚本可在新的空目录重建，各patch是相对前版的零上下文增量。
