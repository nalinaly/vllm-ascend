# SWA计划复用已有RoPE符号任务

前一版单独增加16份SWA任务，完整两档状态正确，但CSA长短8:2回退1.050%，
计划总核时增加35.175%，关键交接仅长档省1.425μs、短档省0.140μs，已经否定。
见[首版完整结果](../csa_sparse_plan_split_20260929/RESULTS.md)。

本版重新从7b296153对应的已测收尾融合副本复制baseline/candidate。
仍只改一个Sparse文件，但把原SWA循环与原rope_cs循环放进同一个提前任务，
压缩计划仍独立等待Top-K和该任务。原plan+rope_cs两任务，变为window+rope及compressed两任务。
两代表档两侧均16+16份AIV worker，避免首版新增16份；所有算术、保护逻辑及QK/PV主体保持。

提前任务沿用rope_cs的min(ceil(T/6),16)个worker。SWA按8行分块、16 lane步进，
T≤96时ceil(T/8)≤ceil(T/6)，足够覆盖所有SWA块；T>96时仍为16个lane。
压缩计划保持16个worker，并显式依赖提前任务，以保护同一token的64字节有效位行。
新增RoPE输入仍为已有freqs_sin；两部分均不依赖Top-K。没有省略校验或把工作移到vLLM入口。

最新AscendC A3 SCFA的窗口/压缩段处理及pypto-lib预展开窗口输入的差异，
仍见[首版来源说明](../csa_sparse_plan_split_20260929/README.md)。
这是一项PTO任务合并/提前实验，不冒称Native已使用同样的两个任务。
两侧总核时都包含窗口、压缩计划及RoPE符号计算，不能拿候选合并任务单核均值与旧plan单核均值比较。

两侧两个入口依赖解析、candidate完整PTOAS/CCE/link/load通过，Ruff/shell通过。
生成代码确认没有独立window/rope_cs第三任务，提前任务无Top-K input、保留RoPE输出，
压缩阶段显式依赖rope_tid，QK/PV源码主体与baseline一致。尚不能据CPU通过推断收益。

两份Python源只读。15:00正常auto提交task_20260929_150040_348018519899，设备1已完成退出0。
128K/B16先baseline后candidate，8K/B24反序，CANN9.2/mode2/atomic0/det0，
每侧5预热20次正式计时、独立四窗DFX及八类跨版本完整状态零容差。
记录完整CSA/P95、计划加RoPE总核时、交接时戳与未改任务波动。按8:2决定是否采用，
有收益再补尾行/padding；不重测Native或扩到七档/模型，生产保持7b296153。

实测长B16 CSA 964.810→972.711 μs（+0.819%），P95 974.160→985.860；
短B24 CSA 941.392→922.634 μs（−1.993%），P95 957.800→942.900。
8:2仍回退0.257%，不全局采用。计划加RoPE总核时145.255→162.215 / 178.195→203.195，
8:2增加12.147%；长档交接18.335→15.220 μs，短档17.930→18.065。
八类完整状态零容差、同图重放/保护区及16窗官方join/worker覆盖通过；
所有组均0/20超过1.05×P50，P95/P50为1.0066–1.0221，未复现大长尾，不据此宣称EP16尾部已解决。

Sparse算术未改，其核时下降不能作为独立incore优化保留。短档B24虽有本轮收益，
目前不据单档结果新增全短档分支；优先转向已发现真实early断点的生产链，避免在小权重场景继续扩测。
不补本候选的边界、Native、七档或模型；生产未改。
[双档结果](RESULTS.md)、[取舍](decision.json)、[含全部计划工作的交接证据](handoff.json)。

[新私有构造](prepare.py)、[差异](candidate.patch)、[来源](source.json)、
[CPU编译](compile_candidate.json)、[静态调度证据](static_evidence.json)、
[运行入口](run.sh)、[收集口径](collect.py)。
