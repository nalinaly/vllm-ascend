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
与Q_B20候选完全独立，不叠加它，不改当前生产。16:22正常auto提交task_20260929_162257_18683029171，设备0，现已完成退出0。

## 设备结论：不采用双head合批

八类跨版本完整状态零容差、图重放/保护区与16个官方DFX窗口覆盖通过。
Q反量化核时长B16为21.505→24.192μs（+12.492%），短B24为32.438→38.954（+20.088%），
8:2加权+14.012%；短档四窗范围30.422–34.457对35.747–41.010，完全分离。
完整CSA长964.179→977.284（+1.359%），短918.781→922.982（+0.457%），8:2 +1.179%。
P95长988.000→995.400、短934.660→941.380；max长988.840→996.680、短940.140→952.480。
四组均0/20超过各自P50的105%，但这是回退，不能以没有大尖峰称为优化通过。

不采用，不补尾行或扩测七档/模型；生产保持f4861832的逐head、Q_B24和七处early。
两head合批扩大UB工作集且保留逐行gather，并不保证整批向量处理更快；
本次只证明这个完整候选回退，未单独归因UB容量、gather或竞争的各自份额。
下一项[整块Gather](../csa_qrope_flat_gather_20260929/README.md)独立从原基线出发，
参考AscendC整块索引，消除每head逐行gather/TMOV，不叠加本候选。
[完整结果](RESULTS.md)、[四窗核时及query链](dequant.json)、[取舍](decision.json)。

[源码差异](candidate.patch)、[来源](source.json)、[CPU生成码核对](static_evidence.json)、
[入口](run.sh)、[采样与分析](collect.py)。
