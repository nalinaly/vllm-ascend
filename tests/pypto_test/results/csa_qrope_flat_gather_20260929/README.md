# Q RoPE满行：展平整块Gather

基线f4861832，独立冻结包v2；不叠加Q_B20或双head合批，生产仍保持原实现。
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

16:39正常auto提交task_20260929_163903_24675961128，16:44确认设备2上running，max-time7200。
私有包源码与运行入口冻结只读；设备收益及跨版本状态尚待结果，不提前采用。

[源码差异](candidate.patch)、[来源](source.json)、[生成码](static_evidence.json)、
[复现入口](run.sh)、[分析入口](collect.py)。
