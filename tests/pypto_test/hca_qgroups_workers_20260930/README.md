# 四组Q扩核与gather、同步的联动（2026-09-30）

生产算子保持0742f07c；本轮三个组合均未接入。保留Q核内变快的候选，供后续任务组织优化。
CANN9.2/NZ2/atomic0/det0，正式第3层权重、BF16残差，128K/B16。
完整区间含mHC pre/norm/HCA/O/post；PTO使用npugraph_ex、dynamic=False、inplace/static开启、SK关闭。

## 改动与对照身份

- `q24`：四组Q的Cube由每组4增加到6，总数16→24；每组仍16 heads，M128/N256/K256、stage2。
- `q24_gather24`：在q24上将长档compressed gather的AIV wave由48降至24，观察与Q反量化的并发。
- `q24_cube_sync`：在q24上给四组各6 Cube开启sync_start，其余设置保持。

Q反量化仍每组12 AIV。所有组合保留TaskId依赖、算术、量化及allow_early_resolve设置。
最后一个组合的对照是q24，不是生产版；没有将两个实验的独立收益相加。

## 完整区间结果

同进程5次预热、ABBA五组、每侧10个真实设备span；单位μs，min/max/mean。

| 组合 | 自己的控制 | 候选 | mean变化 |
| --- | --- | --- | --- |
| q24，对生产版 | 564.750/609.750/587.275 | 581.250/606.000/590.450 | +3.175 |
| q24＋gather24，对生产版 | 561.500/612.250/578.125 | 568.500/588.000/578.350 | +0.225 |
| q24＋Cube同步，对q24 | 562.500/591.500/574.600 | 569.500/595.000/580.325 | +5.725 |

第二组P50也由573.625增至579.000，不认定完整性能改善。
第三组五个ABBA小组中四组回退，max也增加3.50；新六核组的同步未能转化为整段收益。
第二组base最大值是profile首样本；第一组max仅减少3.75，mean与min均回退。
保留原样本，不以这些max变化宣称EP16稳定削峰。
不扩展无整体收益候选的短档、七档或16卡验收。

各组合x_out、swa、compressed、state四类完整状态跨版逐bit；各自eager/graph一致，保护区通过。
CPU解析、编译及load均通过。这些检查不等于本轮新增了Native或模型token验收。

## 同卡独立DFX：单核变快但整组延后

DFX与上表是不同采样窗口，不能把核时变化相加解释整段计时。
下表为核内mean/任务跨度，单位μs；Q四组按生成kernel编号排列。

| 项目 | 生产版 | q24 | q24＋gather24 |
| --- | --- | --- | --- |
| Q Cube组0 | 78.05/78.58 | 50.62/94.64 | 48.57/100.10 |
| Q Cube组1 | 75.67/77.18 | 51.41/58.08 | 50.12/62.54 |
| Q Cube组2 | 75.90/79.94 | 49.60/56.74 | 50.83/58.74 |
| Q Cube组3 | 77.66/79.28 | 50.45/56.74 | 50.34/64.70 |
| compressed gather | 23.67/33.92 | 17.40/33.96 | 33.24/68.08 |
| Attention AIV | 180.75/183.30 | 179.58/182.74 | 184.06/186.40 |
| 最后Q反量化完成时刻 | 299.62 | 309.06 | 305.46 |
| Attention AIV开始/结束 | 303.38/486.68 | 311.58/494.32 | 308.74/495.14 |

Q每核约77→50μs，但q24组0部分worker迟启，整组跨度拉长；其消费者最后完成也更晚。
gather减核后跨度约翻倍，另两组Q反量化跨度也变长，未释放出预期的完整关键链收益。
这些是时序证据，尚无PMU证据将原因唯一归为带宽竞争。
因此实际补测q24＋Cube同步的增量，而非套用旧四核组同步的结论；结果仍回退。

## 证据与复现

[source.json](source.json)、[summary.json](summary.json)、[incore.json](incore.json)、[tasks.json](tasks.json)。
三个patch给出精确差异；`q24_cube_sync.patch`以q24为父版本。
patch使用零行上下文，应用时需要`git apply --unidiff-zero`并匹配父版本。
`prepare.py`和`prepare_sync.py`只创建新的只读私有包，不覆盖运行中的源码。
CPU复用`hca_residual_reuse_20260930/compile.py`；NPU复用`hca_pair_screen_20260930/run.sh`，
全部经`task-submit --device auto --max-time 900`排队。

泳道目录：`../results/hca_qgroups_workers_20260930/dfx/{base,q24,q24_gather24}/dfx/merged_swimlane.json`。
原始计时目录：同一results目录下的`q24_long`、`q24_gather24_long`、`q24_sync_long`。
本轮没有为同步回退候选重复采集DFX，也没有把DFX中的worker累计数解释为同时占用核数。
