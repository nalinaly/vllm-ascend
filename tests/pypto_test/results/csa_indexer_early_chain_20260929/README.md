# Indexer cache 生产链补七处 early 标志

依据[HCA借鉴及现有八窗审查](../csa_hca_early_review_20260929/README.md)，
从7b296153对应已测整包建立新baseline/candidate，不叠加已否定的两版Sparse计划候选。

只改三文件七个生产者：`csa_row_offsets`、Attention Compressor projection，
以及Indexer Compressor projection、pool、boundary init、RMS/RoPE、Hadamard。
移除新增关键字后AST与baseline相同；不改变数值、cache布局、worker、任务数和真实依赖。
长档Score的sync_start与关闭early、短档Score原策略均保持。精度版和公共工具链不改。

两个入口依赖解析通过，candidate完整PTOAS/CCE/link/load通过，七处标志在生成的AICPU提交代码中均生效。
Ruff与shell语法通过；私有两份Python源码均只读。
15:19以auto提交`task_20260929_151902_390858217467`，设备0已完成退出0。

同一卡顺序完成128K/B16、8K/B24，两档颠倒baseline/candidate顺序。
CANN9.2/mode2/atomic0/det0/inplace_pass=True，每侧5预热20次正式计时、独立四窗DFX，
复用既有八类完整状态零容差、图重放及保护区检查。
重点看完整CSA/P95/max、生产链真实预派发、Score/Sparse启动是否延后与核时是否因竞争增加。
有未计时dummy的任务不做ready归因；不把所有交接空隙相加作为理论收益。
以用户长短8:2口径取舍；无收益不扩七档、Native或整模型，有收益再补受影响尾块/padding。

这是纯调度改动，HCA的约20μs收益不能直接套用。当前已通过双档与边界，七处标志移入性能版三个文件。

| 档位 | 完整CSA均值 μs | 变化 | P95 μs | max μs |
| --- | ---: | ---: | ---: | ---: |
| 128K/B16 | 972.964→965.879 | −0.728% | 984.860→986.840 | 985.840→997.060 |
| 8K/B24 | 935.083→917.581 | −1.872% | 965.200→933.760 | 975.400→939.380 |

8:2完整CSA改善0.957%。四组均0/20超过1.05×P50；P95/P50长档1.0133→1.0241、短档1.0287→1.0185。
长档P95增加1.980μs明确保留，未用均值掩盖；本轮未见异常尾部，不代表EP16问题已关闭。
八类跨版本完整状态零容差、图重放/保护区及16窗官方join/worker覆盖通过。

15:31另提交`task_20260929_153109_1479235100`，auto设备2已退出0。
H127/B3/T18尾行及active B=3/2/1/3同图padding、整数/Top-K/保护区与八类状态全部通过。
未新增Native/模型测试，也未重复共享HC入口等不受七标志影响的检查。
生产两入口依赖解析通过，三文件AST与已测私有候选一致；同文件四处旧超长行仅折行，Ruff通过。

四窗证据显示长档Indexer scale commit平均FIN从288.290提前到248.750μs，
但Score首start 328.735→328.400几乎没变：query系数路径成为主要前置。
长档DFX Score AIC/AIV核时反而+4.463%/+4.394%，不能把本项写成incore加速；
Sparse核时虽下降，也没有改变其算术。独立DFX与正式事件计时不能直接相减归因。
Q_B仍用24 workers，却在每窗只使用16–22个物理AIC核（两侧长档范围），确有多波，
但首start时的其他任务占用随窗口改变，不能机械套用HCA“留四核、20恒为一波”的解释。

[双档完整结果](RESULTS.md)、[采用与边界](decision.json)、[逐窗链路及Q_B占核](chain.json)、
[边界验收](boundary/summary.json)、[生产入口与源码对应](production_parse.json)。
本轮收益不外推更新旧七档；阶段出口复用两代表档、补同源码其余五档和泳道，再与已有Native基线对照。

[构造入口](prepare.py)、[七标志差异](candidate.patch)、[来源](source.json)、
[CPU编译](compile_candidate.json)、[生成代码标志](static_evidence.json)、
[队列入口](run.sh)、[收集口径](collect.py)。
