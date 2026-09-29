# Q RoPE满行：展平整块Gather

基线f4861832，独立冻结包v2；不叠加Q_B20或双head合批。两档及边界通过后已采用，结果见下文。
仅性能版qkv_proj_rope.py满8行分支改显式Tile表达，仍逐head和48-worker，尾行原样保留。

参考本地ops-transformer28f40354的rotary_position_embedding：
rotate_interleaved_split_bsn_pad.h的Process一次准备整块索引，Compute/ComputeCastFp32按calcTotalNum Gather。
rotary_position_embedding.cpp包含该路径，README/host注册支持A3；不宣称当前Native具体形状必选该分支。
Native索引按字节，PTO按元素且A3 TGather内部乘4，本候选不重复乘4。

当前PTO与pypto-lib2164563使用Tensor gather(dim=-1)，生成每head按8行循环：
每行一次1×64 TGATHER，再TMOV回拼结果；低层A3 TGather本身也按validRow循环。
候选保留调用方的每行局部索引，在外层加行起点后展平[1,512]；
每head将8×64 RoPE输入视图展平，一次tile.gather后reshape回8×64，只有一行硬件调用。
不改变反量化乘法顺序、512列逐head RMS、RoPE符号与BF16 RINT，不增加GM中转。

首版CPU解析提示Tile rsqrt不接受high_precision属性；v2显式提供同形FP32 scratch，
生成三参数TRSQRT，与原Tensor高精度lowering一致，没有改成默认低精度或sqrt/recip近似。
完整PTOAS/CCE/link/load通过，两入口解析通过。Vec末端105504→106592字节，
满行3个展开调用点均为1×512 TGATHER，尾行仍8×64；整核TMOV从3处降为0，没有TCONCAT。
这些是编译证据，不是设备收益或精度通过；未修改PyPTO/Simpler/PTOAS/ISA。

仅测长128K/B16、短8K/B24，同卡反序5预热20次正式事件及各四窗，完整八类状态零容差。
看Q反量化核时/分布、完整CSA/P95和Score/Sparse资源影响；有真实收益再补尾行/padding。
不重测Native、七档或模型，不改在跑其他私有源。

16:39正常auto提交task_20260929_163903_24675961128，设备2，现已完成退出0。
私有包源码与运行入口冻结只读。

## 两档实测与边界通过：已采用

八类跨版本完整状态零容差、图重放/保护区及16个官方DFX窗口覆盖通过。
Q反量化长B16核时19.253→18.487μs（−3.980%），短B24 32.399→31.809（−1.821%），8:2 −3.548%。
四窗范围长18.878–19.624对16.237–22.873，短31.601–33.073对29.434–34.711，有重叠，
不能声称每窗更快或已经证明稳定的固定降幅；保留全部窗口与未改Score/Sparse的波动。
完整CSA长973.806→966.754（−0.724%），短927.002→918.894（−0.875%），8:2 −0.754%。
P95长994.720→985.400、短950.860→942.820；max长996.720→987.100、短952.080→949.060。
四组均0/20超过各自P50的105%；短档P95/P50略增，不能把绝对P95下降称为消除了所有抖动。

结合生成码消除逐行搬运、两档目标核及CSA均值改善，支持保留本项。
16:54正常auto提交task_20260929_165402_32761723341，仅H127/B3/T18及active-B=3/2/1/3同图padding，
覆盖满8行与不足8行交界；私有包不变。边界任务在设备2完成退出0，八类跨版本状态、
图重放/保护区和全部padding切换通过。已将已测满行函数体移入性能版qkv_proj_rope.py，
生产两入口依赖解析通过；精度版、尾行、调度和工具链未改。
不扩测七档或16卡；最近完整七档f4861832不含本项，不按双档收益改写其余档位。
[完整对照](RESULTS.md)、[四窗与query链](dequant.json)、[边界结果](boundary/summary.json)、
[采用决定](decision.json)、[生产入口解析](production_parse.json)。

[源码差异](candidate.patch)、[来源](source.json)、[生成码](static_evidence.json)、
[复现入口](run.sh)、[分析入口](collect.py)。
