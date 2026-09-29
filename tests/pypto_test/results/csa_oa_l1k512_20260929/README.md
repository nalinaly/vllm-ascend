# NZ O-A：按实际形状选择L1 K预取粒度

基线369ad2c1，包含已采用整块RoPE Gather；本候选仅修改私有性能包decode_o_proj.py。
生产尚未改，精度版、工具链及任务调度保持。

## 来源与分支规则

本地ops-nn19614968的TransposeBatchMatMulBaseTiling::DoCommonTiling按L1容量计算
depthA1/depthB1，再确定stepKa/stepKb；K方向的L1面板可以覆盖多个L0 baseK块。
NZ入口在transpose_batch_mat_mul.cpp中选择TransposeBatchMatMulKernel与MM_CFG_K_SHIFT，
不同于ND侧的PpMatmul手写实现。README支持A3；不宣称当前Native二进制本形状必选相同参数。
本项只参考按容量扩大L1面板，保持顺序K累加，不引入Native K轮转。

| 实际输出形状 | 原策略 | 候选策略 |
| --- | --- | --- |
| T≤32，N128 | L1 K256 | L1 K512 |
| 32<T≤96，N128 | L1 K256 | L1 K512 |
| T>96，N256 | L1 K256 | 保持K256 |
| ND权重 | K256 | 保持K256 |

T为实际token行数，本模型decode是单卡B×6。分支封装在同一个算子内，
调用方不换包或手选档位；O-A不读取历史KV，分支依据T/N形状，无需按seqlen重复选择。
大档N256若同样K512会增大L1需求，保留原策略。当前pypto-lib2164563仍固定K256。
旧验证日志§115曾试过K512，但当时是CANN9.0/WO_A ND/atomic1和单短档整层口径；
本轮针对当前直接借用Native NZ的路径，按双档真实核时决定，不沿用旧结论或重复旧输入。

## CPU与生成码证据

完整PTOAS/CCE/link/load与基线、候选两入口依赖解析通过。
N128分支Mat最大末端262144→524288字节，L0 Left/Right仍各65536字节，
K128 ping-pong和FP32 accumulator保持；Acc为65536字节，无额外TMOV。
每个输出块4096K的L1面板16→8，逻辑两路GM→L1调用32→16，传输字节数不减。
软件流水展开后静态TLOAD调用点仍8个，不能用静态源码计数当动态调用数。
N256分支规范化SSA名称/源码位置后的完整PTO IR与基线一致，Mat仍393216字节。
这证明分支和容量表达成立，不证明设备收益或浮点结果一致。

## 最小设备验证

17:08正常auto提交task_20260929_170813_2046519748，确认设备3上running，max-time7200；
私有Python源码与运行入口只读。PyPTO88f60598/Simplera54c05095，沿用公共CANN9.2环境。
先128K/B16和8K/B16，两档都覆盖T96/N128；按长短8:2报告目标核、CSA、P95/max。
两侧反序、5预热20次正式事件，独立四窗DFX及八类完整状态零容差。
64份O-A工作及依赖不变，可对比同范围核时均值/总量及首末跨度。
若有收益，再补T≤32与N256边界、同图padding和ND兼容；不先重测Native、全七档或16卡。
若不同seqlen/形状收益明显分化，在同一实现内进一步分支；不使用不同实验包拼成最终策略。

[源码补丁](candidate.patch)、[冻结来源](source.json)、[编译证据](static_evidence.json)、
[设备入口](run.sh)、[汇总入口](collect.py)。
