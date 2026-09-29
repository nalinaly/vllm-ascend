# 静态 matmul B 逐张 NZ 对照

基线为生产369ad2c1：不叠加主KV NZ或失败的O-A K512候选。
全部候选使用独立完整私有包，保持任务分工、K累加顺序和任务依赖。
Native已有NZ的Indexer Q/head投影直接借用原地址；两个Compressor与Hadamard在初始化准备PTO NZ，
不改变Native原权重或prefill/回退流程。ND分支保留。

验收先看完整CSA绝对耗时/P95，长128K/B16、短8K/B24按8:2；再检查八类完整输出/状态，
索引、metadata及保护区。核内收益不能代替完整区间收益，不因使用NZ自动接入。
收益分化时在同一算子中按实际长度/batch选择，组合后还需补受影响边界和整模型token/DSpark。
性能后检查使用本轮同初态保存的完整张量，不增加hash校验或重复无关测试。

| 权重 | 独立记录 | 当前准备状态 |
| --- | --- | --- |
| 主KV | [已有两档](../csa_kv_native_nz_20260929/RESULTS.md) | 核内−47%，完整CSA加权−0.080%；候选保留，暂不采用 |
| Indexer Q | [来源](../csa_idx_q_nz_20260929/source.json) | 两档精度PASS；核内−22.268%，CSA +1.114%，两档P95均上升；不接入 |
| Indexer head投影 | [来源](../csa_idx_weights_nz_20260929/source.json) | 完整CPU编译通过；等价取模解决NZ偏移证明；task_20260929_182621_293310313881运行中 |
| 主Compressor wkv | [来源](../csa_cmp_wkv_nz_20260929/source.json) | timing先行，task_20260929_183127_300849123504运行中 |
| 主Compressor wgate | [来源](../csa_cmp_wgate_nz_20260929/source.json) | timing先行，task_20260929_183316_30384662556运行中 |
| Indexer Compressor wkv | [来源](../csa_inner_wkv_nz_20260929/source.json) | 完整CPU编译通过 |
| Indexer Compressor wgate | [来源](../csa_inner_wgate_nz_20260929/source.json) | 完整CPU编译通过 |
| 共享Hadamard | [来源](../csa_hadamard_nz_20260929/source.json) | 完整CPU编译通过，Q/K两条路径共同消费同一张NZ |

CPU编译通过不是设备收益或精度通过。全部使用CANN9.2、相同工具链和真实第二CSA层权重，
mode2/atomic0/det0、5预热20正式采样；PTO现有图方式，inplace_pass=True；不重测Native。

后五项采用先timing、再CPU完整状态检查的顺序：无完整CSA收益就不为DFX另占卡；
有收益再补独立泳道、原地址/初始化NZ格式证据与必要边界。前两项已提交的四窗流程保持冻结不变。
