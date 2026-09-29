# RoPE提前派发：按查询行数选择策略

基底为46cec3a0。全局early候选在128K/B16两轮未稳定获益，而8K/B24两轮及128K/B24控制档均改善，
故只测试T144（B24/S6）开启early，其余行数维持False。不是把长短两个维度混为一种长度策略。
同一inline函数共享原始算术、尾行和真实依赖，运行时仅选择constexpr的派发标志。
CPU已验证两入口依赖图及完整PTOAS/CCE编译，生成代码包含T_DYN==144分支；每次只有一组RoPE任务。

冻结整包后通过auto排队，128K/B16基线先行，8K/B24候选先行，每档5预热20次正式计时和独立四窗DFX。
基线False、候选B16走False，观察分支/编译结构成本；候选B24走True，观察此前收益是否保留。
性能完成后检查八类完整状态零容差，以及各自图重放、保护区和Top-K结构。
有明确收益再补T144实际筛选和padding，不以未运行的其他档位冒充覆盖。

[来源](source.json)、[生成脚本](prepare.py)、[候选补丁](candidate.patch)、[编译结果](compile_candidate.json)、
[编译分支](static_selection.json)、[任务](task.txt)、[排队入口](run.sh)、[收集器](collect.py)、[时序解析](handoff.py)。

本候选未采用：长短两档均回退，8:2 +1.160%，短档P95 +42.420μs。
性能后八类状态零差异，Native误差/Top-K指标新旧相同；不标模型精度通过。
T144实际early生效，B16仍False，每次16份RoPE任务，物理Worker总数未变。
无需再补边界、七档或EP16；[取舍记录](decision.json)。
