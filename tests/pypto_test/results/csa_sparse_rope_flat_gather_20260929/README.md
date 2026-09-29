# Sparse最终逆RoPE：整块Gather

基线369ad2c1；两侧均不含已否定的八张静态B NZ候选。仅性能版Sparse单文件候选。
当前七档Sparse融合核内长B24 AIV233.282μs、Native223.867μs；PTO还含逆RoPE，边界不能机械相减。
以标准长128K/B16、短8K/B24筛选，8:2；后续有收益才补尾行和受影响档位。

参考本地ops-transformer28f40354的
`posembedding/rotary_position_embedding/op_kernel/rotate_interleaved_split_bsn_pad.h`：
第214行准备整块索引，第261/295行一次Gather整个calcTotalNum/totalCount。
PTO表达参考pypto-lib2164563的Sparse发布与本仓已保留Q整块Gather，但不改变它们的算术。

Sparse生产已有单条TGATHER，不存在Q旧Tensor-gather的逐行TMOV。
差异在PTO-ISA A3 `TGather.hpp:TGather`：FP32路径按validRow循环，每行执行vmuls、PIPE_V barrier和vgather。
目前输出16×64，即16轮；候选用输入1×8192、索引/输出1×1024，最后reshape回16×64，变为1轮。
原索引已经包含head*512，不再添加偏移、更不重复乘4（ISA将元素索引转字节）。
一轮覆盖16个向量repeat，不能说总数据工作量或Gather元素数减少16倍。

不改QK/PV、分块softmax、除法/归一化/舍入、KV分页采集、三槽/预发、任务数/依赖/early，
不改Native布局、精度版、PyPTO/PTOAS/ISA。此项是核内指令组织候选，不是新调度开关。

CPU需确认1×1024 TGATHER及reshape为视图，无新增搬运和容量溢出；设备才确认性能及正确性。
5预热20正式设备事件，四窗DFX；性能后核对八类完整状态、整数索引、metadata与保护区。
两侧CANN9.2/mode2/atomic0/det0、现有PTO图，保留P95/max，不重测Native或先跑整模型。

[来源](source.json)、[唯一算子改动](candidate.patch)。正式两档与完整状态已完成，见[结果](RESULTS.md)；尾块/padding也已通过，已接入性能版单文件。


CPU完整编译/PTOAS/CCE/link/load已通过，生成两个pub_h展开点均为输入1×8192、输出1×1024，
reshape前后TASSIGN同地址；TLOAD23/TSTORE13/TMOV9/TEXTRACT14/TCONCAT4与未改Sparse的基线产物相同。
因此只减少TGATHER内部行循环，不宣称删掉了原本不存在的搬运。正式任务：
`task_20260929_193859_406772521712`，正常auto分配，冻结源码后提交。
[编译信息](compile_candidate.json)、[具体Gather类型和调用计数](static_evidence.json)。

实际编译使用Simpler/build/pto-isa 327cd586，已核对其中TGather同样逐validRow循环；没有改ISA实现。


两档完整状态逐元素通过，Sparse AIV长156.128→152.117μs（−2.569%），短182.357→178.563（−2.081%），
8:2 −2.472%。长四窗范围完全分离，短范围重叠，不能说短每窗都更快。AIC加权−2.424%，含与AIV互等变化。
完整CSA长963.185→964.806（+0.168%），短927.921→909.341（−2.002%），8:2 −0.266%。
长P95 971.480→978.360、max978.060→986.840增加；短P95/max下降。四组均0/20超过P50的105%，
不宣称长期异常尾部或EP16已解决。未改Q_B和HC等任务也有读数变化，不把独立DFX差额当调度成本。

边界任务`task_20260929_195407_4663718485`：只补H127/B3/T18与同图active-B 3/2/1/3。
按明确核内收益保留的既有口径推进，长CSA代价单列；不采用上一阶段NZ候选，也不重新测试Native。


与上游pypto-lib2164563的差别：`models/deepseek_v4_flash_dspark/decode_sparse_attn_csa.py:435`
仍由独立48-worker `merge_norm`加载mi/li/oi并做逆RoPE，第466行使用二维TGATHER；
本仓此前已把发布融合到最后PV，现在进一步展平该Gather。本仓qk_pv核时因此包含这段收尾，
不能直接拿上游不含merge_norm的qk_pv数值相减来衡量纯算术差距；本次没有新测上游。

边界任务完成退出0：两侧H127/B3/T18的八类完整状态零容差通过，active-B 3/2/1/3同图padding、metadata及保护区通过。
已采用性能版Sparse单文件，正文与设备验证的冻结候选相同，生产/测试两根依赖解析通过；精度版未改。
[边界证据](boundary/summary.json)、[采用取舍](decision.json)、[生产解析](production_parse.json)。
最新完整七档仍f4861832，不按本轮局部降幅外推七档或EP16。
