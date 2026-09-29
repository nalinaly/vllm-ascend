# 当前Gather基底的RoPE提前派发复核

冻结46cec3a0对应的已验证私有包，只有csa_rope_sign增加allow_early_resolve=True。
该标志控制消费者的提前派发资格，真实依赖、算术、块数和cache布局保持。
两份包都从完整Indexer Gather候选复制；与生产只差两条说明文字，执行逻辑一致。

历史§214在2dd51f15、atomic1、8K/B16试过同一标志，CSA793.658→793.802μs，无收益已撤回。
这次不是遗漏项补齐：当前CANN9.2/atomic0、七处early和Q/Sparse/Indexer Gather已改变前置链，
而旧试验不覆盖128K。现有八窗实际fanin确认csa_rope_sign仍是Indexer反量化的非early生产者。
因此仅给一次当前长短两档复核，不因“补全开关”直接合入。
当前pypto-lib2164563也未开启此标志，属于接入侧试验，不冒称上游优化。

已有八窗还排除了O收尾抢占量化的猜测：所有final派发均晚于最后quant结束。
O-A/O-B各64份Worker在24个AIC上需要多波，不能把全部启动分散算作AICPU软件代价。
[现有来源与逐窗前置](existing_handoff.json)仅解析旧资料，不新占卡。

两代表档128K/B16、8K/B24同卡反序，各5预热20次正式事件和独立4窗DFX。
按完整CSA及P95的8:2结果取舍；这是纯调度候选，不把核时随机下降作为独立核内收益保留。
性能后检查输出、Top-K、cache/state八类完整张量的零容差差异及图重放、metadata和保护区。
若收益成立才补尾行/padding；不为失败候选追加Native、七档或16卡。

[来源](source.json)、[生成脚本](prepare.py)、[补丁](candidate.patch)、[CPU编译](compile.py)、
[排队入口](run.sh)、[结果收集](collect.py)、[调度解析](handoff.py)。

首轮两档已完成，八类完整状态零差异，各自图重放和保护区通过；Native残留指标新旧相同。
正式CSA 8:2−0.467%，两档P95下降；实际前置FIN→Indexer反量化首start约8.215/5.570→0.650/0.580μs。
但长档独立DFX整体窗口略长，小均值差也低于单样本标准差，因此仅做一次反序正式计时确认，
不追加DFX/全状态快照。反序已完成，全局候选未采用，生产未改。
[首轮结果](RESULTS.md)、[时序](handoff.json)、[Native残留](native_reference_residual.json)、
[反序入口](run_reverse.sh)、[反序收集器](collect_reverse.py)。

反序task_20260929_211053_95322323274在auto设备0完成退出0；首轮为设备1，每轮内部同卡。
长B16反而+0.709%、短B24−1.143%，8:2+0.338%，未稳定复现全局收益。
候选仅保留冻结实验副本，不合入生产；短B24两轮均有收益，下一步先比较同B24的128K，
避免把batch与seqlen同时变化误判为单一长度策略。八类完整状态已通过，不再扩大本全局候选测试。
统一format.sh ci仍缺pre-commit，定向Ruff/shell/diff通过，不声称全仓CI完成。
