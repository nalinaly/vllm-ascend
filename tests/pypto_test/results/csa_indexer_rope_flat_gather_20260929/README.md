# Indexer query反量化/RoPE整块Gather

冻结基线为632dd00a，独立pkg副本仅改decode_indexer.py的满行Gather路径。
最新本地AscendC参考仍为ops-transformer28f40354的rotary_position_embedding整块索引/Gather实现；
不声称Native本模型必选该模板。pypto-lib2164563的Indexer仍使用高层pl.gather，当前PTO同样如此。

当前生成码对每head的8行逐行TGATHER/TMOV。候选从连续8x128反量化结果中，
用含row*128和RoPE起点64的绝对元素索引，一次Gather为1x512，再恢复8x64。
不能直接reshape原8x64列切片：底层行距仍为128，CPU审查已排除这种写法，未将其上卡。
整数索引用元素单位；PTO-ISA内部乘4，不能再自行换算字节。

保持逐head乘法顺序、BF16 RINT、四head pipeline、48-worker、尾行逐token路径、
Hadamard/量化/Score和任务依赖。该核处于Indexer query→系数→Score前置链，
先比较真实核时/完整CSA/P95，不将静态指令变化直接当成收益。

先CPU完整编译/load及生成码检查；通过后正常auto测试128K/B16、8K/B24。
每侧5预热20次正式设备事件，另四窗DFX；性能后比较八类完整张量、Top-K和保护区。
有效才补H4095/B3真实筛选和padding；本项已完成验证并接入性能版；精度版、Native和工具链保持。
已完成的七档严格属于632dd00a，没有混入本项候选。

[来源](source.json)、[生成脚本](prepare.py)、[补丁](candidate.patch)、[CPU编译](compile.py)、
[单卡入口](run.sh)、[完整收集器](collect.py)。

CPU完整编译/PTOAS/CCE/link/load通过；满行两个pipeline展开点均为1×1024输入、1×512输出，逐行TMOV消失。
尾行两个TGATHER/TMOV调用点保留，TLOAD13/TSTORE8/TCVT16/TMUL8/TADD4与基线相同；不把静态计数当动态收益。
20:15正常auto提交`task_20260929_201500_22994431416`，冻结副本后不再编辑，两档结果已采齐，见下文。
[生成码证据](static_evidence.json)、[完整编译结果](compile_candidate.json)。

两档任务已退出0，性能后的八类完整状态零差异。目标核时8:2−15.193%，完整CSA+0.240%。
长核时14.081→12.062μs（四窗有重叠），短19.124→15.566（四窗范围分离）。
长CSA+0.014%、P95+3.120μs，短CSA+1.146%、P95+13.180μs。没有区间收益，按用户先保留核内收益的规则补边界。
H127原边界主动终止退出130：全可见分支不充分覆盖query的真实Top-K选择，未标通过。
替换H4095/B3/T18与active-B 3/2/1/3，覆盖满8行、两行尾部和真实Top-K筛选；冻结算子未修改。
[完整结果](RESULTS.md)、[取舍及query时序](decision.json)、[边界入口](run_boundary_selective.sh)。

H4095/B3真实Top-K选择及同图3/2/1/3任务已完成退出0，跨版本八类完整状态、padding/metadata/保护区全部通过。
已移入性能版decode_indexer.py，保留生产中两条更新后的说明文字；去除docstring后的AST与已测候选一致，
生产/测试两入口依赖解析通过。除26行新增/8行替换的满行处理外，不改变其他算子路径。

Native零容差仍有既有差异：此H4095/B3的x_out max_abs=0.0234375、RMSE=0.002612776，
Top-K集合替换39项（顺序与集合的总位置差7699）；两侧旧/新PTO对应Native的全部误差指标相同。
这是无新增精度回退的证据，不是Native跨实现或整模型token/DSpark通过。
[边界状态](boundary_selective/summary.json)、[Native残留](native_reference_residual.json)、[生产解析](production_parse.json)。

完整CSA 8:2 +0.240%仍保留为代价，不用核内−15.193%冒称端到端加速；下一步处理query链和O投影交接的调度。
统一format.sh ci因缺pre-commit未完成，定向Ruff/shell/diff通过。
