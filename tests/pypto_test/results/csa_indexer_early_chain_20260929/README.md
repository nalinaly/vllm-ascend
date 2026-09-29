# Indexer cache 生产链补七处 early 标志

依据[HCA借鉴及现有八窗审查](../csa_hca_early_review_20260929/README.md)，
从7b296153对应已测整包建立新baseline/candidate，不叠加已否定的两版Sparse计划候选。

只改三文件七个生产者：`csa_row_offsets`、Attention Compressor projection，
以及Indexer Compressor projection、pool、boundary init、RMS/RoPE、Hadamard。
移除新增关键字后AST与baseline相同；不改变数值、cache布局、worker、任务数和真实依赖。
长档Score的sync_start与关闭early、短档Score原策略均保持。精度版和公共工具链不改。

两个入口依赖解析通过，candidate完整PTOAS/CCE/link/load通过，七处标志在生成的AICPU提交代码中均生效。
Ruff与shell语法通过；私有两份Python源码均只读。
15:19以auto提交`task_20260929_151902_390858217467`，当前确认running。

同一卡顺序完成128K/B16、8K/B24，两档颠倒baseline/candidate顺序。
CANN9.2/mode2/atomic0/det0/inplace_pass=True，每侧5预热20次正式计时、独立四窗DFX，
复用既有八类完整状态零容差、图重放及保护区检查。
重点看完整CSA/P95/max、生产链真实预派发、Score/Sparse启动是否延后与核时是否因竞争增加。
有未计时dummy的任务不做ready归因；不把所有交接空隙相加作为理论收益。
以用户长短8:2口径取舍；无收益不扩七档、Native或整模型，有收益再补受影响尾块/padding。

这是纯调度实验，HCA的约20μs收益不能直接套用；当前尚无设备结果或采用结论。

[构造入口](prepare.py)、[七标志差异](candidate.patch)、[来源](source.json)、
[CPU编译](compile_candidate.json)、[生成代码标志](static_evidence.json)、
[队列入口](run.sh)、[收集口径](collect.py)。
