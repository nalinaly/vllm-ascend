# Q反量化/RMS/RoPE：满行双head合批

基线为f4861832对应已验证整包；只改性能版qkv_proj_rope.py的满8行分支。
48-worker、每worker负责的token/head集合和任务依赖/early保持，Q_B仍24。
当前上游pypto-lib 2164563也是逐head循环；本候选借鉴本地最新AscendC ops-nn19614968的
rms_norm/op_kernel/rms_norm_whole_reduce_sum.h:SubProcess910/ComputeRstd多行拷贝及批量向量处理。
该op README支持A3，rms_norm.cpp注册WholeReduceSum路径；不宣称当前Native二进制选择了此分支。

相邻两head按[8,1024]读入和反量化，再无搬运reshape为[16,512]进行逐head RMS。
FP32乘法先后、512列规约、high_precision rsqrt、RoPE与BF16 RINT保持原序；
余下不足8行的分支原样保留，ND/NZ/精度版/工具链均不改变。
原来每head输出NOPE和RoPE，候选每双head仍为四份对应strided store，没有额外GM中转。
两head对应cos/sin/index在外层复制；gather当前仍按行实现，不能声称全部指令减半。

v1完整CPU编译通过，但A3 TCONCAT输出会每双head进行32次逐行UB复制，未上卡。
v2改为输出reshape view和原布局分头store，取消内层TCONCAT；只保留外层三个小输入拼接。
完整性能入口PTOAS/CCE/link/load通过，两根解析通过。目标核Vec最大末端105504→148000字节，
容量通过；没有修改PTOAS/ISA。保留原gather的TMOV，静态调用数不是动态指令数或性能收益。

只排队长128K/B16和短8K/B24，两侧同卡反序，5预热20正式事件，独立四窗和八类完整状态。
目标核仍48-worker，可比较相同总工作的核时均值及范围，同时检查query→Score、CSA/P95/max。
以真实核内收益决定保留；有收益再补必要尾行/padding，不先扩大七档或真实16卡测试。
与Q_B20候选完全独立，不叠加它，不改当前生产。16:22正常auto提交task_20260929_162257_18683029171，确认设备0上running，max-time7200。当前尚无设备结论。

[源码差异](candidate.patch)、[来源](source.json)、[CPU生成码核对](static_evidence.json)、
[入口](run.sh)、[采样与分析](collect.py)。
