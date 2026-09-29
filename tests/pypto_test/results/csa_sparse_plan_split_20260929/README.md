# Sparse计划拆分：把滑窗页表处理移出Top-K之后的串行区间

基底为7b296153对应的已验证收尾融合副本。两份新的私有整包，只改candidate的
decode_sparse_attn_csa.py；生产、Native cache布局、工具链和算术都不改。

## 已有泳道证据

复用收尾融合candidate的两档四窗，不新占卡，官方raw join和worker覆盖通过。
Merge FIN→Sparse首start的四窗为长B16 **20.26/12.88/19.78/18.10μs**，
短B24 **16.46/15.70/17.56/18.66μs**；均值17.755/17.095μs。
Sparse最后前置FIN→首start仅长0.78–0.86μs、短0.54–0.70μs。
关键前置是合并的csa_slots_build_valid_qk_plan，其worker首start→末end长8.26–9.26μs、
短11.52–13.30μs；末end→FIN还分别有2.86–10.48、2.12–4.44μs。
因此整段不能叫纯算术或“Sparse没及时派发”，也不能把计划结束确认成本忽略。
[全部时戳、真实前置和原泳道路径](existing_handoff.json)。

原计划里SWA窗口长度、页表空洞和padding处理只依赖position/seq_lens/原始页表，
压缩索引的合法性检查才依赖Indexer Top-K。当前把两部分放在同一任务，SWA也等Top-K。
候选让SWA单独提前，压缩计划仍等Top-K和SWA，Sparse再等压缩计划。
增加16份AIV worker，不能只因关键链局部变短就认定完整CSA获益；正式8:2和P95决定取舍。

valid_block_mask每token为16个INT32，即独占64字节DDR行。两个阶段虽写不同列，仍在同一行，
所以压缩阶段显式依赖window_plan_tid；保留原行归属，不允许两个阶段并发标量写同一行。
生成代码确认SWA无idx_topk输入，压缩阶段有Top-K input、两缓冲inout及显式SWA依赖。
设备泳道还须验证此依赖实际存在且执行不重叠。

## Native与上游差异

最新本地ops-transformer 28f40354的A3 SCFA在
experimental/attention/sparse_attn_sharedkv/op_kernel/arch22/sparse_attn_sharedkv_scfa_kernel.h
按isOriOnly分别处理原始窗口和压缩段；压缩索引的边界/地址验证位于
sparse_attn_sharedkv_scfa_block_vector.h的GetRealS2Idx/GetKeyGmOffset。
这支持区分两类输入依赖，但本候选没有照搬Native完整流水，也没有移除索引保护。

pypto-lib 2164563的decode_sparse_attn_csa.py仍将计划合并；它直接读取预展开的
window_swa_indices，而本仓为直接复用Native位置/页表，在PTO内部求窗口页边界及空洞。
这部分额外工作仍保留，只尝试提前。没有新增外层展开/复制或要求vLLM更换cache布局。

## 当前验证范围

baseline/candidate两个入口依赖解析通过；candidate完整PTOAS/CCE/link/load通过。
Ruff和shell语法通过，私有包全部Python源只读。
14:44正常auto提交task_20260929_144431_322320312386，已确认设备1上running。
128K/B16先baseline后candidate，8K/B24反序；CANN9.2/mode2/atomic0/det0，
每侧5预热20次正式计时、四窗DFX和八类完整状态零容差。
收集完整CSA/P95、计划总核时、Merge FIN→Sparse start及未改任务耗时，避免只观察等待转移。
有收益再补受影响的尾行/padding；当前没有设备结论，不做Native/七档/EP16扩测。

[私有构造](prepare.py)、[差异](candidate.patch)、[源码来源](source.json)、
[CPU编译](compile_candidate.json)、[生成调度代码](static_evidence.json)、
[运行](run.sh)、[收集](collect.py)。
