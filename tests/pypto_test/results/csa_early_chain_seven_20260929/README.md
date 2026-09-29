# f4861832 阶段出口：补齐七档 PTO 与现有 Native 对照

使用已验证的冻结整包，不再构造或编辑算子副本：
`.cache/csa-indexer-early-chain-7b296153-candidate`，
`PTO_CSA_VARIANT=pkg:dsv4_csa_indexer_early_chain_7b296153`。
对应生产f4861832，包含O-B激活L1复用、单行HC收尾融合、七处early生产者标志；
不包含两版已否定的Sparse计划提前，也不改Q_B workers或长档Score尾部门控。

七档：128K B4/B8/B16/B24、8K B16/B24/B32，B40继续退役。
本轮复用[已完成的128K/B16和8K/B24](../csa_indexer_early_chain_20260929/RESULTS.md)，
主任务task_20260929_151902_390858217467在auto设备0完成，含各四窗与PyTorch profile。
只补其余五档，15:43正常auto提交task_20260929_154347_45284811883，当前确认设备0上running。
本轮不是优化候选重复筛选，而是三项已采用策略的同源码阶段覆盖。

沿用CANN9.2、NZ mode2、atomic0/det0、同一真实第二个CSA层权重/合成历史，EPLB关闭。
PyPTO仍88f60598、Simpler仍a54c05095，启动前核对没有更换公共工具链。
每档5次预热20次设备事件，独立PyTorch profile与四窗图重放DFX；记录全部P95/max与异常样本。
旧七档不保存完整跨版本state，因此本轮七档自回放状态检查不冒称七档新旧版本逐元素一致。
跨版本依据为已完成的两个代表档和H127/B3尾行/同图padding，不代替Native/PTO或模型token/DSpark验收。

Native直接复用[最新标准七档](../csa_native_inplace_seven_20260929/NATIVE_RESULTS.md)：
npugraph_ex、dynamic=False、inplace/static/SuperKernel开启；核内诊断只有SuperKernel关闭。
两侧采样来自不同任务，不能把累计变化归因于某一项，也不代表16卡模型forward。

采齐后收集完整CSA/P95、Indexer与Sparse核内及调度明细，补充新融合收尾任务的工作量，
防止沿用旧报表时漏掉已不存在的独立hc_post。统一复制window_3原始JSON并按七档命名打包，
不挑最快窗口，不更改事件，来源清单保留每份PyTorch profile和原始四窗路径。
当前未完成七档，不提供拼接版本的性能表或下载包。

[来源](source.json)、[五档排队入口](run.sh)、[每档入口](run_side.sh)、
[收集与融合收尾明细](collect.py)、[固定窗口汇集](bundle.py)。
